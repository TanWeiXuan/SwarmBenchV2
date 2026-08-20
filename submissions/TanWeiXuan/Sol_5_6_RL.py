"""Sol_5_6_RL: Opus control with a tiny learned tactical planner.

The geometry, flow fields, safety filter, movement, collision/projectile
avoidance, interception, and gunnery are inherited from Luna_xHigh Opus Breaker,
which in turn is a minimal tactical modification of renj1ete0's Opus 5 V1.
Only an occasional high-level SCOUT duty in ``_plan`` is selected by a learned,
centrally observed 91-48-48-1 value MLP (6,817 parameters). Inference is plain
Python, so the submitted controller has no model runtime or external weight
file. Every acceleration, path, safety decision, collision resolution, target
reservation, and gun command remains deterministic.

Learned planning
----------------
The 38 global features pool variable-sized live drone sets into time/score,
counts and survival fractions by type, transport progress/time-to-goal,
ammunition, pressure, projectile risk, and pursuit geometry. Each candidate
adds 14 features for one live scout, 32 deterministic scout/target features,
its previous six-way role, and role duration. Dead or scored drones disappear
from the pools and candidate lists immediately; counts are normalized by game
limits or the sampled initial counts. No padding or fixed swarm size is
assumed, and an empty live-scout set is valid.

Once per simulated second the MLP ranks RUN, KEEP, HUNT_TRANSPORT, HUNT_TANK,
GUARD, and BLOCK alternatives. The deterministic planner supplies the best
feasible target for each role. At most one non-RUN override is accepted, only
above a calibrated 0.40 softmax confidence, and it is held for five seconds to
avoid role thrashing. A dead scout, dead target, or expired commitment cancels
the assignment safely.

Training and experimental record (2026-08-21)
---------------------------------------------
Experiments 1-2 tested allocation-profile cloning and PPO. Experiment 3 expanded
the set-aware PPO actor to 28,432 parameters (including two extra 48-unit
layers), but it reached only 42% against Opus on a 300-game gate and was
rejected. Guard-only counterfactual models also failed held-out tests.

The retained policy instead uses approximate policy iteration: at 140 sampled
states, authoritative simulator copies rolled several feasible tactical duties
to match end. A listwise value MLP learned those terminal preferences with a
source fallback. Its final source is deterministic all-RUN, demonstrating that
the larger PPO actor is not needed for the measured gain.

On 250 new paired seeds and both sides it scored 301-49-150 against Opus:
65.1% match points with a paired-seed 95% interval of 59.9%-70.3%. Across 600
held-out matches against all 15 current valid community controllers it scored
509-35-56 (87.75%), above parity against every opponent; the hardest were
Gemini (53.75%), GPT (63.75%), and Opus (63.75%). Independent 500-game gates
against Gemini and fixed mode 4 were 52.7% and 52.8%, respectively, so those
small advantages remain statistically marginal.

Design rationale
----------------
Instrumenting matches between the strongest existing controllers showed that
the single largest source of lost value is not the opponent: it is
``OBSTACLE_CRASH``.  Top controllers routinely lose a third to a half of their
team to the scenery, because they steer with short-range repulsive fields that
have no idea where the free space actually is.  Every crashed TRANSPORT is
five points that were never even contested.

So this controller replaces reactive repulsion with real planning:

* ``initialize`` rasterises the arena onto a 0.5 m grid at exactly the
  clearance the scenario generator guarantees is traversable, computes a
  chamfer clearance transform, and runs one Dijkstra sweep from each goal.
  That yields a cost-to-go field plus successor pointers - a global flow
  field - biased away from tight gaps.
* At run time a vehicle chases the furthest point on its flow-field path that
  it can still see in a straight line, so the coarse grid path becomes a
  smooth, taut trajectory.
* Every command then passes a forward-simulated safety filter that replays the
  engine's own jerk-limited dynamics.  If the intended acceleration would put
  the vehicle into geometry within the next ~1.2 s, a fan of alternatives is
  tried and the closest safe one wins.

With the scenery no longer killing anybody, the remaining losses are all
trades, and contact destroys both vehicles.  A 1-point SCOUT spent on a 5-point
TRANSPORT is a four-point swing; a TRANSPORT that touches anything is a
disaster.  Everything therefore runs for the goal by default, and SCOUTs peel
off only where the arithmetic is clearly favourable:

* a TRANSPORT cannot outrun a SCOUT, so fleeing a pursuer only postpones the
  trade - a SCOUT is sent to meet the pursuer instead, turning five-for-one
  into one-for-one;
* an enemy TANK is worth one point as a body but is holding five rounds worth
  several TRANSPORTs, and at 1.5 m/s it cannot refuse the trade;
* a round already in flight is stopped by any hull, so a SCOUT standing on the
  firing line costs one point instead of five;
* two SCOUTs keep our own goal mouth while anything is still near it.

TANKs hold a firing station covering our half, weighing each shot by the points
it would deny and by how much room the target has to leave the contact disc in
the time of flight - a TRANSPORT inside about a second of flight simply cannot
get out of the way - and once their magazine or their half of the arena is
empty they go and score their own point.

Attribution: the proportional velocity-tracking steer, the closest-approach
projectile dodge and the closed-form constant-speed intercept lead are the
common idiom of this repository's built-in baselines
(``swarmbench/controllers/baselines/common.py``); they are reimplemented here.
The grid planner, safety filter, role assignment and gunnery are original to
this file.
"""

from __future__ import annotations

from base64 import b85decode
from heapq import heappop, heappush
from math import cos, exp, hypot, sin, sqrt, tanh
from struct import unpack
from zlib import decompress

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team

TINY = 1.0e-9
SQ2 = 1.4142135623730951

GRID = 0.5                 # planner cell size, matches the generator's own test
PLAN_CLEARANCE = 0.6       # generator guarantees reachability at this clearance
LOS_MARGIN = 0.45          # half-width required for a straight-line shortcut
COMFORT = 1.8              # metres of clearance below which routing is penalised
COMFORT_WEIGHT = 2.2
CHAIN = 44                 # cells of flow-field path cached per source cell

HORIZON = 1.2              # seconds of forward simulation in the safety filter
SUB_DT = 0.1
CRASH_MARGIN = 0.12        # extra metres demanded on top of the lethal radius

FAN = (0.35, 0.7, 1.05, 1.4, 1.9, 2.5, 3.14159265)

RUN, HUNT, KEEP, GUN, BLOCK = 0, 1, 2, 3, 4

TACTICAL_RUN, HUNT_TRANSPORT, HUNT_TANK, GUARD_TRANSPORT, TACTICAL_KEEP, TACTICAL_BLOCK = range(6)
TACTICAL_INTERVAL = 1.0
TACTICAL_COMMITMENT = 5.0
TACTICAL_CONFIDENCE = 0.4
TACTICAL_INPUTS = 91
POINT_VALUE = {DroneType.SCOUT: 1, DroneType.TRANSPORT: 5, DroneType.TANK: 1}

# BEGIN EXPORTED TACTICAL VALUE WEIGHTS
TACTICAL_WEIGHTS_B85 = (
    'c-j<~hhL9x8-`0mTa!>JGnLGwzRz`$nNXobDiWfMtTNM9+Nq?WVI^cG>HAz)$Sf5pD=SLKCVoU-?|*RLpZh+K^E}2sI8VD`BOp^l'
    'Rd91~5TE_WeD%mwy0xMTG=t}$ci0@>lb8s_=78^)WKm7mNAZmt;r%!>_6uby;AVOqdl~w@X@HDT?`eU*GDnz97Up@ZV{OO1u)IA4'
    '(8C7upKXFcYg%B+%vI1kV>{<*rl2_V5*+b~6$>M`m1`VyMXLlq@o=U)^qQ<ivqQX~MR`7M^+^!(hD4*!+vz-3b{l0w1h@{(=FJJS'
    'ut&H74qAVJsxOb>Nh!KiXnu+kM)|V#<-T|;c>)I=O)GckR>ge-$6>Ll4R19s6IyL`czd8McHa|<#?OzD$6T9=m)Be2^^BVsTGxwj'
    'uWl4_WM{F7Z9Ca5N|UtMBvJT){rFy0mn+{6V%Kp4QGIio_`Zh`Z_?1kPF@B310K=tesNfpbDNb-Yhiv}KJA=8Q9O0`2zI_~79J;O'
    'gP+DkF5cLV{jy%bvmsTO>RCvdAD!?`W)!adSxwWgtHStUO5k~79M(*6l9*_X5YDZYk~DA@HYu3!f$zOZrCbh8<o;60<r18DIu^PT'
    '7UL|lS$NA`otG??M<?x6J~z4#b>5uJ;f|?baOXw2!<TsgOLXy&!F#%qa{_+%NvE8>zGz-n2D^6c6$jsYEgUmFLOX6$2rJ%u^O6--'
    'JZXp$i*v4{l9OzO;I$5h?wmvme;bmw|5-RSX&F^oDsV#WVDwvGEdK23#jUza*kx!w$&^^&``qa`D|#u&Khl7)>B+oB*Mm1rcmY{m'
    'yTKvOlCu^q#`*rr6x5u6^>06meUFUAb0xJjDgOsK)h!dv!iz||I17&M@qwV%GjP+pbQrN}CA(=F@zjfz+!FYi7kkfQkHsvynJvO<'
    'BLg0OaRzjaXak+~gLv8~35T32lh~}xfwGc3iW~2S=N69yBc<y!VnP=bHcUd@zunmTxgC5mbb+?qbodzckwSw<gK_yvaY1ef)?erg'
    '&unu9YrQ8>PnF_018pwKHN!B$6%PM(=Hqh<$uuYqKQ;d*&rDymigsnan%hwSVJxdC9pJHDv$#*@JWT6Z&$kMrq4Zmt<o!Mudi?kk'
    'e0_INI=O8b5B&HVsx@v=%-0qnsxyW6yn2jJ9s97?;b(9$O;?(qu>pIX+X5d#wP@nsdDL>dJI7t0PFJq(z_`Nh7@V^QUinASl+Gif'
    'Zmt4)n8<T6^<xc(TvSST#53C4V2bZB_80eK$H*j}k<<aTgVT6k-CJ5APNW8n2$1R?5p0soIHq3<wFGy^qTyj^uAsyBc82q`Ohr7r'
    'bpXrH-j82w_R;8=6_D+zkH>>YiGgdQxvIS{kNMNcpXn8ao`1xXMqY-W2KlroLX$mpPxFm_U#Vt%I2>7LgAUGTrGCbv==63~GJZ6K'
    'zlP_Md#5GzdDl)aE0l2i=ypki<x7~!T3njEAEf;&VDBjbo%Ltq<i_KoO~odN7q#%bs}GI#lS5@`3fMlI%JaXu6SQBXom2AZ+sNId'
    'vf>5hHjd;opDuxXQzYDSS;Aif{-e6YHxx2S6%8)dh&drg*)4Jsq-39=<u`Z1sblUuOS&0-I<E+$0}AQCdDq1!1ABq4dj((eF6Q}T'
    '7vf2qCdrPEZurL}M5w8i^08NGoOw=Gx@X2GxUl#>NB;7F?z5f6$n6h#U2=b1BD;*V7QUpopFd&4k8J)idI}#~o=zW3jCkh_9c(#0'
    '9DOPRc+X)UjydYhRholH{&6F0__CYUY^lWo=N|A_m*-@tHVeP(d5ztUA{(w8%D#hb_>6ZE)W2E;3m5Ij`=|Xdla|t`qsPH5eE<jC'
    'D8*B`)@*60Ot?yuA5Toh<)2!_Th1Tgeq9DOEgXjRbDZG9%tAhQ@;2OAvrpV}ri+pyBLDHZKYkr?h!*-Ma^*I_cOiDX`&0^<M=E3G'
    'yFA+HHJRHxtN4boGW8FL<<awEY(L%3qO+kraf{Xzvgr0zh&A~rj%=~P1*I#&yD$v59V(R0^*7`1-;yvRXf02;bYBc66*&HGFpQ8+'
    '<HT>T`9jndHtTf4Fq@+|cvrCq+0SW3R3^Uv62q_NB+*^70jT&riwus<#kw<Q_|id+dw;qJ8aKUgeeOv*ruYfZZqQ;oxgVgA+W@sy'
    '*FfQ8pm4fy9_zQi6aRSX!M8I<$z8Ub4Da^9fRQP9Lh%f5?KuTiWbIJzRtzVtS^_qEy=ZC5EAd8|3HQ_d3qL}~k>mbI(fOr3SWUf#'
    'Es7~D9Z*9LWlvH^=p4Q@QHwsv-4s@zDyKeaS>k!SiK2hEdpK~78#WZka^AoOSen=-Tr6G4_Z}H@KS?3V82MAly6#kXY(1~-8pL)J'
    'j*9D>Zjp<vGG2aqh*rr>z!_sTVP8uoKQhn7V^%LDD#7#E>y#V#7au|+*^N9W;W5nqW`b8^9@4B6yCD9;0gSjcga<}-<JiSL$o5-3'
    'XZ0R~+4ZydlcO^Hyl@YyqGt2>_(7<e5rz93hk;qyNZR>&IbJTl!ui<^^xi&BT=;Ph4X|8{>kZW?$0Cvb;8QA4e?ZS0S3^nSU3T&s'
    '#C01>#TBEbB0N{ceTUPyuZ$ilg^8q^UJDOq<?@PN`@#FlV;W<YDqOkW4@b;NrmQ<A_($2GVs`)kgx(91#e^eaG}G?i8>Y>H@}{-a'
    '|5h@M-nR>08Y}T(r5#*8ZZ2FG0<hfTCpoxe<L$v`P~7CihkBW#YwKI#=RP;?wP!GT9b3l6_sqE0uQn)NbptkzT*0dKH>C%sM2MY{'
    '-lFE^e`kF?o;{{Jvf6wvu9WPDid(w4M`bL%(5VC8RhF1iAI7_9#&C9iC`R9F6_%}>0e9Xw@=Z&$ZO2Ve=$^+(M^3|gD}7GV>c&I1'
    'm(ix#S9y7B8Wl-i3mUB%WYSH*1Jy5Konj6Qe)0m`7VV{Ot%WqdW(>Tkih+QO3KapRCY(@O0e^!=!sN6QFgqdzG!Nxa%RS(4jn?Qp'
    'GgW%&+e!Z0SOt~StJrjPHr!}C!0Oqj$)bBZ`quUZi*p%Vcx536<bI*Ry~X&S(OKwNZOn%G=WVpyw&Sk-eZcmQA#{e%fEAgB9DlzA'
    'Mmwu;i0f|d>=?t3x@4gsV+6?V7>KF<HP~sZi$9XoD55KkD*6FFes6^q@fz?vxEr64jlp;N*P!j98LR1(P}9cuw9_b6GT6+Oql$;4'
    'Ufo@hH`Kz9oN(b%bql7g>&e3>&wxG62cY*+c|0oTFPv>WDGc;mz^_M6=DVTcur<7jG{0ox)R{S;n6Vk!(teQra(C`GsseV8I6!k!'
    'ck-a3)sU1b2S=YyptGmGlbS&^XgvBzMTW*)X`jw}Kb{t5t3Kn{4CI0z4p^mJ$mdTlz=viEf^K>Rb)2zciO*o15Oj=RSe5aoW&N@1'
    'm^G<dSJT(%G@RGlg)`0evbe+^TiU*Ye&q(tvlzf0^Zn8N=Nvlw=Ybe<x(bbFr{nObU{pQoCf)M&wXken4YVf?XZIy`*!yxgmP-Rr'
    '&3P~u4>IA9eHy&bP>+B8x<WG_7J~hR6ljfT6SFpNA!Ri=_LJ(<iz)hi>28cL&gckUEl`KBLB@DgKS`qStS>%!<ss^1zJ{IHA9s!!'
    '!5$mkXxZHb*zN5ZoE(%(LlpO7+D}_aU0hE#icW@?OD!OA7Lm`|pV0r08L3T(;O7e_unt}ld1@uz%TGkDi3*%(IuCn)C$PQUS6JCx'
    'ByNx!1iM$w7l!?)Bj;<`wC<W5pIT%onzldS>b(2nh&kri7*K=_|MvB}i&?_N(#=@1S%HqGWs0*~-_lvxVmR_33hwUhfYESO@_6D;'
    'GKjoHP7SH-duFUSeUbuOzHbzU)a&BnYn|}WH5gY6t%dB@I=s-PKwPKy5jG4p=AFUIdCv5Fe*9Vm;(99y^|ocg!M0z}eo_m=%{<w-'
    'Bp2dG-NE1H6VYR7BMm-Kh&ztzR<zGh=a%)o@#*;i{QUScp)qzJgp5AHGG)QEPIH!EHtn^<vFxohG&Nqhc2bMS9B+l=`uBt@hGG2S'
    '$w7>>IwJ{Horj^5Ot5;I0vj*uO+)^x5KjaS;}a(qpk2;3e)D4+hndQw!j3zz`1muB8Ge%v_R*lYj>&AD*IgKEzn1G8PQmn_@wolv'
    'Ae`hGFZ5`zr`;!O(2q(nBxWkfdjEt!zaE2+`E${x=QK2Y_n9WXGUV^?-@yBn(O7HVn@tN<x$RFq_R$T)jn3IPbJRgNby9{uEWJk~'
    '7w>|Dwr=2Td<A|i^<vZACn2szPqb^@!sa{ffx(#LZ1lQ}&8PIooRj~$&L@Yz^%z1jcePpOtnPU4V@&OCp*TF(RD3bEAF6S!kT$FX'
    '3V-SH{5e@Ht^UU^hwh+U{Y`Q~uS(c+{R!>PD2A+NZybLnj#<VW^-V9}3EwtOaat^LzQ9iEb1@~cl6A|@k>cil!r&wYocDUWG(D)5'
    'ei)@wo$C{bTyu|x%uB+j8#VEFsV{r<O`?^~0m96+Yw)hS8htIylh(KOWtn2tiiL{B^k`lm(b^<h^uHbf1#?#8wCiUi@t?HdZ%Q4E'
    'n%0ZA#C!7ybtC>|r-Z{A7jpXOyJThji0=NhfCW>ZV(&i+!oKe^Fd<_Te>YTwd!OIY5Zin~@_sa)KeG->O!e3pld#D(M@;G%hMR+p'
    '#?SD{DqpWO5;qBs(sjFB1m8##TyV33eLE$zvQc1-2_<y%`!t-ntAS!RJMkVJO>(`lk9?o#kzdRZJgpQ%m$Xw@Yl#DPtj&ijcfxUD'
    'P&sxVxD-DBolIL^WRRxi3{c*=k5jH$!LXnvpvF}0dODx3_=of8=<n3$xF?h*T2RF}1J3L}c)a?y6bQ_+#eENZqO-Oq)fqOEb#EI!'
    'S{(x4l-EGWy1k75{z>wyI<Sx5%lFh0q-XP-KzpM)Rb0tO^NfA?R^O~VNwOSARcUg<vVP+5HQGYnA{VkXzrx1rJ^6WpEYDF+7K+9j'
    '37?&pv-FWA7w8@0L#NGP*hv>&a$1guC)L9%Ar5OtYjU_n9DRCaD4uT}OHZbpV%$<7>i^CXhT7F}*2Z@FsnmocZz8HMjFD(q2hyVD'
    'yEwe>c)b7TF8Fs&rWUP%&?J=6_wo0*?d~W({yZB#`CX&^+bik%^Q)qK+Bxby#1po8dSYY9SnLKm7?S=**g5DI-0xq&I*ZkC`IAnt'
    'Tz!l;9C}5+Aq6w`g@QvL2iCO=<Qr<<@HA)=HCse-S{l&vq2c0$+J{`HRs^^ERME)&>e$iS6OOyA<Kk@!9MgPFC@3n#x0f&R(0D5f'
    '{<)r#7QTX-^yBdL>uubfYJvt$My%OwFJ$fUV}qa;LFYjaes`$=C(Q}RKH3>z>k!KE!U*it>`n6$0)+Fs66kqi1I!HEhG+NNqa;nC'
    'qBChf4?lBO?Dn=N{)~6Sg4X}UlC|gXaN#6D`Q{s;NnQdPqxR#hG40Ya&l42%RukX7$`xZxMw5@8JzDN)Aw!#Iq+gpxdgeVb_s>DN'
    'U2_*kY83F{6}t58Xa^jXrb?16{)2wKbI|s?Bjh_<;*772JaY0C`nqYgI5O=r-28PAdbQr-<@?L&htmd*s5=c$Pwu1X`7yX6D~L*U'
    '?0L}qDdM;%d8qm2mBjH|5@`IH#Vyy;$@^y_$b<?wHQa<f`<nB`MTw&I+K1?<tbnC=J0QjW5iIH*DmoYX)Asy<?0I4m)qVQs`SDZO'
    'w@(<gDohd&rd|el3r#*+kVL;Jp0o02f`$1z@kQT1LTjQMKHU|-Qv<aBaqpsV=*t<QC1xF$<etKdk>A1c*<kKzH5r<Y8KCr}7y9Wq'
    'lIDS_JYn8vn%YaiTh{3`xOELhEKU?<16;Vqb~c;c-wDA3Lj=u=lh{vr8%$m$uuFwDCO`B=o83Eb%MAghEa}5i!*$sA$Yz?<d>A8o'
    '4`BD8!>k?fK%Ah}3o>ThvCUn+g#8-qxu{u#{@%{!FXPLg%hM0l&m5$$e@24F_Dz^+9*UGckrl_jl*A5yEn1E*V5KxWa7s(y1F_+h'
    'R?-Kj`P_!zch1xL2p{Z!K@H{~D1$ov=X7uEJ$yIK7T0{xMTeq%oEdctZtOOMHB(mN^{dxG)4>OajFaOv19x+HemHN`_)5LBPQtYT'
    'cj3R@Z|PL)LAcsz35RYd;m@SiobxLcV`i;~e$Hk%bIKo@G2tZ!%>4l)r?x=!6E`lY%o7bY9)YvRE~p9i!VVV;&b}qms*Z3x)!IT%'
    '3pR3F&(-AfEk=MltHEYLA#JX$V$EwWpt-JD+&S4v46$j4xJ~NxYxrk6s(FHbh74rA=TC*`-#RcdQh{I5Y?9xSKw@txd8c0%w&yp%'
    'mA=FH`|61pKYs>S_tF`E|MvoLbiR$(9N&UtF7VYe<5_p7FKjsBh}pgDIBjXO&~dR2Hftgdp1Xs7-}={?`wnBBt{)aWD8e)LTM=&P'
    'z&-CN*njmq3>?=9<F94ljBN(w>ZM9sRFv7?Wrxu7fGwHrcH>}G8_A@$Yjk7AVUz}6;4R6kap+S4mutO2r<o=2Y`7GTTH3Og%1JmQ'
    '$rr62w$OTN5v>mzVT$zuD%^X7h9|_*t>$#fzMsI?EA>!q?ky_*aR?r+N`SPJ!+6Gn5-5F~44qHApmF1WkUGLs`Z+Lz?c#%QeIG;4'
    'jxB-a;bHh->|1#I8_{cXC`3d#@dgz))EO$zg`c)didI%jUU(+Zrl1}CV}u-^4mPULEG^+#D;)7yz6kT*XtBvTCH!N*j8;5~6k`s('
    'rnCWpbf(ulJhk&N>=><s0qY;~i5OqDOn*YthxoI(Fch;C{is~`H91TkgYuiw5O<8huLmxXt9LK78}$JC4v7LSi4iCDQD-MlE!_IA'
    'KU;U4Cx`b9WU$|uoVq*1E>AD4)z!wntJh)te<o;tejF4xP2|-t(}V`K1Mv2OK<184;ymeDHuRMx6SFX`>9vXVLUX~*2=Mxusba&k'
    'W*n~Ai}i(3@JY!GuTOsk;(b%tE*mXQ|E4Lb%Y1<IvA20clNWz~W6awYd?K;-G}H%1VK=#2(BJ0A3CrEFU~MjVZ|lo*74K7AO#;+3'
    '>x!C^sr0dPELWC>;O&jCp{RHjZr_v%r)L_oY3MS1toWBc-f8EitH{UtU!k*KjLGi&F<4M^3SO#3;H~F-QU2eXT)4w%mofu!lM4H9'
    'PeOHL4K6Krhe^9USUTGiC#N>TfQLsw_2+op*XJhad+(y<iRVE-yM_ne?SWh0X$p4@#^AGfU2wc($dA$^1%qEJacSf@{xHec)@@h1'
    '&~e9;<xVQ&uiZ>1CwHqjJ;R4hLObwFPav7~>%<?~4&qdu(d@r$3cvTR2Go=lhL#u6xXFX@*3nW5G~F+R*Eqtr5o7qysckT0xDE%J'
    '`s1n}Az)+CjyolayxLYmb^T}3ufgvq)X0Q?tIDCUM#36TLg1jGKbxL-A~d|ej4zkT;G|M-tm)MuT1QR5+So8~9zKUP&h&*tOZ(%;'
    '=Zbuy_zJ8$;SD3I52DUQ3!JPnANzEL(XkN^!2a4WYKz;1x!ST2@x+%B7aqWZh3S;Bpjxa6*+SpD&cMjUIbu#VfMfM}T<6_`(-NXU'
    '+20MucE3P}t)GftU##a#Q|Dp#qCup6T?rrVI*)J5Ho<}7-k2qy1!Wu4>F9<kia&WoXwIk=E80JDfX(!Zp%*Oqz>wp@tF6n)%X$P<'
    'g#=3T7G-1Cve8sws)VCQ55)_kcF|t7g}g6l96Ei~!#hV0V|w8XJpLtyAC(rd@0Ekpjs3{UJPJR)i=+e5PsHKUJ>ssv1S|XXL1$?r'
    '-gz^I7xs6>tXl%V@L0?l7f0guI|I;i&Q;E_GUWaT)!45l4T23_sAa_?GOJlgsn4sae2}^1xpoO!@>;mEdIkSlv;&L{03sYZz-`n|'
    'STHjgPtR|sJGay*`nVRKpHdI)%k&`shz#0wy_Wb)@_`YFXZYZ+E8zARvD@8Gw7aMuzY~;kXR;1EE_UU^ebvygbtR`Mb_nJs>MTqs'
    'fjc|T!|z>V#Q%(!L+bV-o0SW02o<T_@W37o9Cy3}9=!;Kk0X1c!GG!`=PSpFI!8F|{wX%Re~bEM?}gVsVN@tiCzre&Fxk|HXKcH{'
    'Z<Yq}L!0mP{Qf;yw&x~A-k!;ax(?%*2s7|ABv9JW4joy8s9}30PbnPB^;;eZnFB_mZ?__T_s)R6pKqdJ9S3=2d#lJ7hw_(wrz!JR'
    '9?0CeNOp^q(cy6;jkDN?EraXCbE;d}Aa^J9Raqg`Z;-=3;%(CO{6hWn`|!3-6^sc;6Lnv2{O73!m^L>MW(7^d0q1T&+vUYL%d3EX'
    'K46$QaRdg0D)Qbi2OhiEnCi4ufJWbCo$Ll&UmgR(V;iV>+enwL=<~^Y=ea3jBA)xC&(_im_ADREm%j$!<J=U|ZmtxzC_I2hqZE9u'
    'm_(k+qfoV220MyI^XyHj(pOeh(#h_}A=bGGgX5zxYGM&y&yu6l%H^WGfgvvFyAjWsN8k-Ld6r+@3ifW3K!{V}>#v;gy!9$bpQ^&|'
    '&U$fI*C?FbyNsS6>;(P0f$V?fp<rX|CVKjh=1ScebZ_f@T&$KThCVi8?;h&-xaTj}Gh-^RZ^|Q?C&6qt$ABlR)=*}V4J3^lLV=b='
    'V!T%$esQu0-PVr6K}&m6_IWqyMAJO{F+2-O=6|6Llls$=&GB%n@;vb0SG4X*H7x4ilM^56aPli9RzEh1Uzna2#&lQXeEDo%lcm9K'
    '6YWsPs1aP(US`Ga72=?wH!<c~0`-2mAGPN`q$jhKaMh{NJYmXEb~i2HI<MUrSGj<-ruN1oNdZ(5{~P?*m6Gwrc|5jFmc*@>FI>Bk'
    'Drj2`#TJ7D6jWFNahGESg9{as%1li>xoa@L(b<6|S%z$<<cl8bf*>k*4raEk=k(Y>-jTfFpYt4|X&wsvqO)1_x8Dt`2HE1MwEmp0'
    'EMUGkf#q&)h5Jb$1j)`Oo3#eCsl807tXoEBzrEyZ8|AskLKdUPB%x3Fdzw9L3=TC-XN7Sa!Dx;=&0RQ!?^L<)f&TkZVsns9D#x*V'
    'QWH*a2;<`eD>>>y4^&w+lLqR|K>vgf;+yc^qFRwMXGeD95v8TjORB-eLCg7%d!p3est}{JRQYJ}Cz@^L$@}GlDExZ}b(_5#GUEzq'
    '!%J)Nc9a5Ud0vv<>+2}=Ih`u1zfIv!E#|mzjK+9e5DdZddV%J&6BOIUkdZq8FV79&H2YTk;SkQxcmyh}Es<*bY(uAI>tIa3^W4K='
    'CYrv`;AbB$l491sy)JV|G_>EzQ`)z{*qhtfIdeAl($zw-xK0>)<CFM%=rmsaMTW{B+2i0iON0w^B}Vh@c|*-6R-5aEdy5~@_6&bv'
    'j+lgv5ysrn6hQSxV}+Zmzl+JM)Un3pAjn>o$Ex}z6!cUNpM7ij#|1Y|Z#@tHwGGA3QbV@%YlQTN+fZf3TZp}Sh97B-LHD^mg_2db'
    '1x=eW++})Pq(?0{=$kxyU;V`LcA0$ly&O*;<d0?6?a=vm46Yn96IYq65(c;HvrhSP@yW<is_T_NLuI9KMEX=%xV;<q^O(zj?^|%b'
    'w-U?yEM@6%89tXim<xZW@$?TLK->1RBtS`lr3bE2>W<lHRh9~uR}96)PgJqXWdLk!nZ<`2ow;>In&>y;HO+n2i_Gd4Vr1TJ9CqiJ'
    'm~$r@lfqv@`9WXk`jyI$W=!DOAD=?y_AJrccQD$@ZNoAC(R|H11#f>`gGLJ~D6FUlYHBXPA`czu@{mt3AykKAwkhF_`itU+z@1>A'
    '5RV_Wl%dh{&$M%MI+S$D(E_^^zMZ_8V}hIib+#qR=I-ao4}%aKc7olFe&cJ5XJgZ_ZfvD`M3@=9loGE$6UN6(g=HS~q*joD^^feO'
    '`%f<xJbF~a&5QlGTKKo$EB3&yfOvd=-kT!67)d5?DT2{Ib76&AD77!>kJIEj=){rL{OMpGsjV^Q_6J*V`B8Zenl_%ts`lq67tJx!'
    'V<|eE3`V6tXW)HCJWpP^56T0V;<IWUGT3gwXX7O7>U~p;Y>I`~F+JE{GfnzJ68w+7IlM$uo4#Mq!-1;1MTfjhDEcv$+*<pfLw^_k'
    'J8F7)#fvNaeEAJpq`F*UaPbh9D4gLfY9YMZH<Vwb$&bIfww5;3qFA^5FKEwCX4TwNLie7baK*z4pB#TKUie+eo$uN~w5b%ISEY)x'
    '6;mWfugX%cW)*2m7V}d1ui}2gI{vXNgkndzLY?CxjQ`?Edrf6<Nv(uqv}SRGx-(zbS7jrmAPj0Q7Z2ZkMssaaI9{VCX8uXSa5#aM'
    ';dZ=kW3}Xd$WdH=_p&s`M1?JNMhHPMqv@e{A*q$!Av^1_wBCJ(NK23N=lm5YYj1(=p5xf!zqkA}_knb*h8y2>x5sq%BU~9}%#}^b'
    'd^v9=KFC~;{Y&SQao_+Pd-f<!*w{glu2bl#>su=MIgy;|Ca}-KVKgB$2i68Wz-c)}aN~ywpL*iRHg2m}GHIT0vqX+8Ey_f@D8?rj'
    'XJW3b3aLHo!45zC#3NlXG+<gV54xJoKkV0ta<^QuD&Lmho|}PX1uncg_@yw@F`1UWNE5P;x$v3JP}uG~2(CYq@P_oUkhu}1<wN4x'
    'P(Bhzs@#TT!};jdeh*qN*U{*{2heYu7CsGIAu6ZDah_iiG^+;EWvLqW8`voXScKwN_d;}3@#dzYA@Fj!1+6ZzK(FdGICK6Q-ri52'
    'U7d==8<&>xqq;<hi#<W7BD>?juD)0`=?k692>f@a3*;GnjfSe<<g?Ygz&<^M%sb|Q#m@|WA8(2+yM}Y&h6()Oa|oJE$-s@bU3j^{'
    '0A4jsgLC^0!F5C1z+w|Y=8ZUU+QKO4(2tP58qk|J-nx!^+;ga?*G`tbKNL-(<v|rE3318e__K2aeGFfWTRTFyo4pb!88kuRpQV)k'
    'vIxwsg@TLXDae-X!?<b{ZnmAwLpGJt%qOzwX|bx@?3^+!*O`v_4i_b@HZ|PCB||(ZIY(O~_TV{74`I{vV_3Us2oAQM#D}!pxxa-T'
    'PD@B-w{$Z!*z%Kj;~`<m{zzKbx|i(72D0ph8fuwXO*(gdV4L=O)X3V)wg&Oo&tnlc`YhzFvtQ9_PbqJRXohS1RB1evij$*13A23d'
    'NL8l?Z+ccCO#OA4RtmpFy?Gb#?4x$>YY|VyeFyUouUOo0^Cxvvnuh1ohT{P-1V=14gSx9pygE4?HI{7TVR?sO(;pAPB+y3`zCWS0'
    '9n<kv^$Lj)cN`P!*RWFVFz$YDBZj1O(7r7#u>J2p|E)^J7ruJv(=wf1m$nE-9Sg8?UlFHjG(qQ{=d|NfhHz@fC>Fieu;*ecT#-A4'
    'de&US_t#Rv?Nl-tote$E4GTqYw`tV3L7ugZtuSfnS6UF_PDcg^82lzjI90BL*Y6&uw$@rG-nNk^b{V3d%o&)U{Rs9Sc7$C&e^X~K'
    'FE;3t2-~*@z$@qJbbJk>qiBLkzxCPtzoj^`K%K8RSb(0(Y}m0yk^Wei(uONZJgMUk@YZZR{_CYUWL%eU!lw&r<hpU_10OoOMT15!'
    'H-`+LPH2pA!?t7Df{wi+78K{xvH=FT!fXaE*}j{{C&lB~n#bU3K@fF*22M&3g6a|nIy9q!`n-t5Y4Q1(xppTew^~AG-*_HfZ~@1f'
    '&*E#>H7e!~)576fzR})<pFF9x9i#8=<)1aNJoRu1$iDZ%^X*+U{r-BqUpG})^e^Al6CdDD&t0s2aT@RHaYzg_h@fPtHYJ6I!=9qs'
    '&~sCYR3=iHA9y&UtjbD&?}k_v<|D=%+oHOIDm=R)hrX+osIN={+Sbc(Y<oR?`0uVHaDO)5O4P?yBmTn!m*%qZX<xMK7LVg<EyUQV'
    'JNRMTXx7b{0IU0bA!+(aRM4C$RHxmiqs8YrvY}Wg?P-P;BU4#U<_j3SO{T4mXL!>vO}y=Yh9;j4q@#DM(7VHfzFypmBX;egQ^TWS'
    'S8Ff)Q+AN-TX%!!++r9vF^&7k9KwdV_n~BSFAhF73-#aWqP2W79(?I3adk5n61VH&JL|KQQgNP_{&yJM_H9Om%y4Wo=*}DNWwHO~'
    '_d=zv4D@tZOZ^q)usWm${si^LpLaJ1zk{C8yojfw!r3#Jxoa)997tg8t(Q@$`Zq~Gb&{@Io-k3rLdb}COolxLJnX-T%2MJX?w|?o'
    'u$d-U<lccDitp*oGorjzcS(6xD)%?g!c$9+<J*nJp!T@G*zh72H@5u;9fAr(S!l7-u6*I1({MJ*jb`(;gJIBw;q2*TEct#Vmauvd'
    '6yM1wrKfGQ<WvN$n|>J@=lJ7)b4Fp<c%p*pg<RCyo9p)IqS>rG(XlHZ{7%VY{r*t`Jqe`|onPSBrpb)k4{_9$4%!*+k0HUTG-$3S'
    '&Xp~LjID7fSGx?pg@kcZ|2??Z4neZAOq@`#fJf&1`!mOtdEE188XIwt)8=bZ`O86A7kN)8JQ@b2?z4E1>n`^B*bNVz3NQCIKF*=<'
    '>ml!t2^mF)lc)J^w7mOJyxywMV|S^*^QhY}$!3_~yJR}~?d^iW@n6|kD;2g&Zey*-6H=L7jhBN9=(n#GAL(8T_$!OLn)kB3=~Wo8'
    '<S@Uvu%33w8PnO7X|zK+l62nAgMg%B(c|Y=>3H`TYQ8lRulw$#FVUJ@TXUFi<uAgz>%-}CODB6s%jj`mIokhTfaV3IqJ3Trui5Uw'
    '2c8`j<JwMxh2<|$Xl@X?0w;?Dz1t}HxHk(*)!3$f635*ghPSZ-7WdSW6f|k^lU95DbX7uMPhZ16e^W@_sX~nMHpKl|nN-BndE(Y@'
    'keF}`AM7yTv_WggRq~d;ZSZCcH9&{@Xnxbvh2n+j!lC&qA#JEOH(2!J-<M?ZW&LbOl+%XZabLx(fq!Z2nHW4aF$qRocu2R(Hi%A#'
    '$MHdfbCQ8wH|X4-DwZ907C&Yib7`O!d@d}Z(f&F-`*OChuptg9&W{v(^`e<k_3*B*A-alp!K&RFPwtz9E_V{dpvNJ2msNxU-Ev#m'
    '9(El5sTT$NL_mH>AHGuMhZmMu()xaj`Q)QzY^ZvUX_*0bTI*ukC6Z(pH$lR|@oc&%jJFpg<A^k6F1R=bVh<k_DZc?+y1s(r!$s))'
    'cL1GA`hfN4t0gzp0V98Gp<}<jS*6zmP}w{S0&X6qiY7T&9I7EKFB*y}=kC&m@_zJ3-H?^mt8vDQzT<E93@hK4u#ul8#PhP}eqwI*'
    '2{Ec{6(;}bh96{J(U!RRlwzrc>bV{aznp1e>lso}w8s7=*Wvu$2w}bKH1@E5&WdR$nWsNjc%l*w_1$#wTkAgRR+j)t;d|g>Qz%8<'
    '-^k;g|I(7KB5*ZU!Y@;zxzD=weDLZkA)tE<94gS~$Jd78#JF-XuF{YXM!WH*-`3n%u%EWJ3>0T=R7V$+c=T;)0W9oCdxQ0b3+`9M'
    'O-l8cX8x4j7c^6%&qBPnULSYG%L}k|HD%YiLv)l3Ysb5aAC1-V;@7Lvg|&NFa>kY{lup~$n%RK#DG7^&a2{||6=~so7*g<oUd#%G'
    'C(D%azwl`KF>wTM4H?em@|C#cR$sx&%$2Mwitw^`5ubWC9Wx)?pvP4QF(5dMCnSsj)0BUli@n3vO4BG?c_TF@C~#7`5vD%efv=UG'
    'P>=K*)NQaIu2icA?S^$+rZfPO{<Ftqe>Jl1vz})LtP%J6Sc?q}M}@CP4`SYnp<=A~g4P(ClCsSm>HX(}c;@#TSg)f>5B=;!wtETN'
    '%IXAp=?(swy^R~Srs47SSj?Ig4c3*3D6KsUPg6G13E=^&M7cx2<`djiCPiMhoV?H-{Y--Rj(2zTREZ_OZ7QU{@eX&s3}Ct3qk@+7'
    '3>tnef#F{aBr#Ku;My_kdBEa2uB+F<r(ci#>*6Fq_n{7M5<b&`r^rP&t0;KNNtk`;mtZ{87Bb2-`4^_qF|SpU`OS&K?_V47W68hm'
    'HKj?=RNICgJI(mUwAm>6o=C%!X7It%kMKb%K^5y|borhYG}`XstMVQc;x?F9&e|!c*v?|}#=+v4TdGihSP66fOvLMFJH*E^Gx*tr'
    'Gf-x-m(owju+5bemYLI=Gk5K#3f&MsvF-w``yGj4mzv;0=X8u-ok<_U<l&&S5Tr7@1i{!5PjA-5A3i^X5${K$+`OT@DSIUM(|JyZ'
    'bMx`a!-Zng7f;+1xRb<hOX>LMU_mB*DD2#JlPouC;>kaoSm&{n%YHu)m$Ys}Mdu4b_nPyPgsL~7CFG+_%`I44xC>0QOfW1}mrdoy'
    'q5jTr(mxx?d8dX_($G85oS}>#$AzI<iY5=<=D?0Vt8mS=e&{vUi6_je5%Uh-7Hhl^dIa_upOrWh=lpU<!*@sNzCnLs>HOzpvEY<2'
    '{8bYBPp^Ujiy!in`Yt*d+)pg%d4Y<W)H%WP8kByjhkjQk@w1UHNz2IrZ?9X-c^<KBy}A*G4eQTyUS1ZOPK0B|>l`ZB9?y9n{J?Jf'
    'F`@B=BTV@H2{d*`QQte)1%)rssQoFMoFYDgL{^cFSC1Cj)(7JwyZLCX^&QBNpuu(yx7BKiu48OiiT3c}jyQI0QQ?FGn=$-wDGW^Q'
    'i(eFbvenx&oU<^MgO;3Q7sH#>^Ph_<$w#4&T{=dz`Qgc~{nYl;kjy-`P(P!QJg?7DT43D(i=1^)@w_XmuYDtx>D7}9oYTntlZ4xU'
    '%)|{Tc@T8`y4aK(E^55=K(|W`P!e($(n8WG<L@kfVe+r`_3b%dGF_zW8hAMC2!Fr%gIx^9qr2A+n))FFR{UClt0!C+=AJvnRSQD!'
    '&#Chy7p4j-G3RmQz+5Uf=_%$6cR<;bsj#cFPAK>Gpqy#VaMo1~tY+7smr=PeWsD7Sry@3wS&7DP=8L)p>g;NJpML0j;Oux4G=35z'
    's@MGyw9K=pbbATDP<;X`UO7r-pZN=8*Ih<~oiE_frzkexm4eED3i!MGA-r?HhN^uO_;sriPBvH}X|77gK$8b-Z)VTS?=0kGwPQ4@'
    'tcG5=yNPe*7vZ39zIZ)w8chga#J7G7C;Q;TqG5Fty;rNCTY3d}e%NBp&FQ4Ure^%}Lmb^6T>&2Zzl%rUJb8I2a#iAB@^?N8@>6G!'
    '3%AqN(_5+MIA2`7>0gc;7qc*=7CL?#<JG*)TpRxdACEf0W-g6XHS{8#obM_PYzu>@+r!{=*hA^r#n~L|BgNUzy7S>1w&<OCmo7>B'
    'V#}@D^g6PN?AL8ZU$Y&2`;0rjSk)7jrLLp)ZVGG{_Jp+k7r~p)O9V^B_h9y28<*}M1HI=Yz(jR-RN7f|;q>(|^o-Gm4?FJB(3#gU'
    'ZT=-`<+zVfUZ}tqZ}w!ZTjijpeUF|h_u%&O6W|kcjF#QB$Kc{rsZqjpx_DoeBMlO9)x~t0=e3(70w?gDj{ea9#8$SQrYXsi1fWT+'
    'Cd#PnC*3XKTvt>Fdmm|W#3n;{Qhf^^IA_t02b%0|;353fG)HmmK-S-23Cdtxu}01WKS!xy)B4#|*nJe<kZ&d@+XP|sKS%KTdkM}w'
    'Z?#P^GXb;Ca2TjC6u0!A&POdrVCDN>c*vkjn0R;qEFQH|^p!~xm-RkMKMgY|;q^^kby$O6%2k5V^zOLwV?LF>NX6bau2HajCEROi'
    '!`S2U6$ia;1AUCZwvPhoYhHllc}sZcjS{*yVF;QhCt~`g6jnS^jUF$J__c;Jl)mZ$dxbi1Yjx$h-JemCM4Jqy+i`}P9rVb$NMCCW'
    'D^@pLgVg~G;d9^y*t;wi`W8>7#^YQ0yX;<yxfdZ;gc|Yi%?_O0W{2&!qiqIBVsX?{Ega>d%r8CX@#8n$$2)z<#kh^P`DpGAtY4GD'
    'OG8bmp*e>AGEUHPaTQBe_Tk-8FUUhiPY7y@#h)8a!2vFlth(%jlbVv@;mv%|US>ww&a${}zAn1Q2VuwKJmi0lwb<Vc_od3h{Dqa^'
    'lss2(PyHnfIO_rfp3H(1Dyzs}XcxE6wT9dMed*;Dc{X&BW9#N<hUCZiq~b0cg~rpABS!4_$%g)33uQUckVfwLM2ZK#Q;N4K813)H'
    'Z(0&iq`#mS=MJhxZJZZ=2Hph~(vAy`=%y0IbxR_#Y*8StP7Nhlbv<Es^Gspb&iC~4_(;4xW(c@YAw0I1NL638(`ArH2dNEr%)XAZ'
    'J{<@DFG2XpXbW#j*vDVu_fgv3@f5c$2Fiwn@qpPfqEa7UcrzmGU#?e^!+jH;6Hci2yctfYPQ!>T5^lMj#7&c~QSA?9^p?*iAFDJR'
    'H~)&Dnj1*9fBN#6UVq@v(nmt8yB*GTT1;>EBW-y&h3sBuiw|!-hqrr{am?;CTxorbd(BIMPaRUw3w=$t*;<?-?&QtG`ol!oTV(Ur'
    '1!k0L;~2er;+#z`7~fY9reAD?C4*-0C_Q&JlMN8EBq^AaV?y(-I%wioJ>hb2lW4Tp2lm)bhORw&g7w~T6iSQeMoAy+hdX%GHy?1#'
    'enXB8x#BrU#LdTN&;j$qxEy}4!pECf?rk&P>HS_X#aLEo+Yi<^sv%&{2w`(*rcKkYkHYfNS$N_7aB;?iq1fn|#((@J_~u*!UViP%'
    'y?(hugL;-YV_6Q#yf=ogWf5q6#SIgeO~r*%e&U0%hWIl$72H=P(pDue`n2rG1uNGv=w_D(X)jE0_>o>5`TIE3e!oaM0j@mf(HIOV'
    'yaQuK#o^rBddRbhL!H%FrRpH8IXMTctS@r!I&};*JAl#oAY}gdC}svK(w5E?z9G!Qs)jUafXO<3TyzD0g`A+V2L$Y>w4^(>3S8T-'
    'nf=|Y*<r&T*khFe>))AUZ=YmZsWFgGsG9K}Q*AUJNRn5VdSdIrM0y!91e4mW`C|PTyp)<NE(u!6$@kZ>URozD^E?Rec^d_-74h4m'
    'b@=^QB=pTZK$}AQ^S58O1dIQ?$#;7_rpjvIluK7>U9}w~UQl3-&xk(`WwO%81$@kN99i01iARSIpaDjHw9{ZVJFn`+wU2%SPIM-T'
    't1_$ou;Nqc<~Y@%8`lRcg%1uEEbJ(PvHCATj3^cVZcyZR$9nSPei>}5`h=?HU!$~fsqnm;rTF~7GrAesC1h_<fEEYh8@)dY{lE4m'
    'wvB~n3kGp>paqY)jl#Ra>!hm}E*jRwp`WiguHP{R733<&Z24NgGSYx+6)J_6hCGyy9RfX4hvI$RICM9CD;Rf+ET36!ga?kvv%|2d'
    '=%pRN-?kp)Jx<7xsuMYU`x;c=p9L=Y-O#B#iN{(zhhMrk#jVG-h;hqQ@vz-6mXH$Xo|l3Dj!&fZ5ywdO%Oo1QTo=9!a3@K(%JP>M'
    'B`CG_q3$cA*z(&&IA9*ekCk0;#5HpkBO|%XJ&6)k`ZBfl<IvMaoIUa+hxV+MypJ2kXKfb!<NYH1uvd0`RPZk{$#aBiw?1THHJc7f'
    'PvUXANHW}73CCwF6_afS;Faasoc2=*`#s!?F$>?rbsJAU(DG3n|J9LqH0x5fsR~|-ek}H0auJI27Kp!>FGKhFk>sbWM_UylafxS!'
    '*sJ{`O*9SS_{oP@D{U64*H2~5@6WL5Qasx>EW#M`U^wRgh7Erx;>O}USgD!!uN%g3^p67+l)VFe9e3iL=b0QcIZQZl<hP(O?kU|V'
    '+ylLhBJrb~A||TGiV8Muq+GR#KgVw)r*Y<Fm~Fvrqg;5Mbul}bF5;>4$8xAkF?j#IKr#seULW?A&J;PpluUgdJ;npxB{Ye?!VKto'
    'wgP8<dj$c96|j9lr+E1GQZ8-G!9nWL7+`m}Tz1UAZP|X7LNiOn-9Lu%`pjG&e0L?k`m2jodKS>~=P<sO8diAKe5G#pGQ_g&auvQw'
    'gM@X1LLjp!9Tv!Z5PwK+!H?!RZnjD%>v2n9p3Z&@T-yv6zifcbW#ufW26OGQWq4$XF^*IWg%#%c;=tw8`CM!++ZCT<o3B?e?PPEI'
    'Yc7imqqMl?9}~~F?d4qGan$3B4Xsum%im3+;lv~tKIUt~xxux<<8~L^rewlNTS7U`<PJ=s2wv2CE>1^H$hk8bS04`I^Y7j9@5MHl'
    '7IR)SJ?@5ax7|toOgN;pJ9A0qN@}gmK-FQVs4C?)B=y-uBX1hB{LVJou%_7d#Ag?plPu4+<>8X*G8uF$d;medJ@9m|1-zhg1f6uq'
    '6OV3RMo#}adihpYo_B94p8on-EKJg)|8maIi*cK|exVsp9sWU>-m!z{$Vi~2K?@dWSK-|w=cH4!E5#(&%h<DYD|n{vCjISyU~<t8'
    'o@f<=`5kH4>+3?eb>0>2M<M$qD6;96wYaEz3!K1jq-mnWb&XHKT5d4Nt%=~P=jLJ6h4p0eU7K&z=<ufAL-^)*MLao94aJGbX^*CJ'
    'W8y@<6h$a=T#ZeBdeQ=$H8|;Vrl9cqEZzTX!K0>zLFwIQF;zDZRpZprPir6sl}w;l_tfEdvJ>y<{tNs^ONEgg4`^}pCsBIw9}@yk'
    '!C{kraH-r0jdrg`O)U)^7#~iXO3NT@%1Er3s=+h0mGG-^DP1w%N0qUnxPCxi4!SnLw$G%?kYFDxBu{%lGfEmk=cFwAgq`Not!DVQ'
    'Oo<1YsIxfZ0yIj3QRl85_MiWkO6mrJX607a3mpyb4liIUy$g_kOGB7^x|$Y@Ur+J6Pho#uh9Et*T6h?0K&RJg@{zymNUv`KHihmJ'
    'yBAuM)vs=BRiw|GPg(I0+28cRM*{1Ij>I!wek{3=$^i>K(0R*ns^Q_tM#&JD8$y2uhw{^%3s66G45~GEQb6S(@#g$0KH@(Yw#7}y'
    '3zI*RM@a}+S>B@6)1Pu0D~u1lbe;ZtWsc)`7Zk5q$d+5yNlo`of#)MHh$=n)@8z1fZA}`6cGtwWKc>*cj4NF8vzE;g9>Jpn7eVir'
    'CuUS-@w{zgF;8z5syyqY=pBA&sDXHE&IQW;;sJ{Dw1uj7XZhy45(t^Dj>GG+>4dr;1`k?>L`THHY**g@GD=$afbqs>eSG?H7tVAx'
    '<Iy#M1yj3^cfMlF0TFim-}@1~*iQ<Y8VY!C^GEs`GE1CgI*ZozQ04<phVZAV6S7qcV8Gm+@SijvhWCqv2i_X&t#S}*fA?dH9WhkW'
    '%N8})jpDlA$}Dk<z~3gPL?5NwV)WgC@X4T#cC2is%a;+SeZ9?#ch13%glZgEFoU&1risa3R#?=02X>Sn6IyzlgXyr3ns=KCqsJn4'
    'k1OM}f7$EIX@IVogJ_Z0K8`otjn|jgf}P=XJnaiyed4OrXnz=M-}waYd+KSa;vVF!H^{o*Ou=m201p513)22hqHT`5@ypp1nAf8K'
    'RexC1;Hh1Zud)ZvIYx5!l&j+J^*NxGHyYdhBzU3Jm2&1^73<uMc<9VO&^2o_DJVsY=cc}<H6e~@u~&wR7kr}nU9)N7D+PR|_Xd(b'
    '9fanOLxmE5FYc!hMq#4_ZYwC|_Zi8!Z~ih-K_OS@KEj(e4Lbr)zJ{a0aBVzLcM$4)CSqKBrBtRxoAzGm#pCBXV`@YyHT~PIcbhzM'
    'XIU7pe>W9xP6M9dFqP*%pU-X*sj#L!K@4xSVyF2byu2j~hhA4^nbHP$`5=S6E?U3~uc>Gus<QOwd`Os`0lC|=#97W$pwTdm108P('
    'FDIRVWr3G4!_<_gwYJd$oqcSXs=_a3z7s7?j-$akMYMN(Mtvi;lBc#0+U_k7)D2=ds<O}cthuRp{ryS4SyzrZGVfrC@d`mpxryrr'
    '{)81XM{u#<zZ}UP2R&skd_AXuKR>X?>K+?7T5}a!C5K>di+b9~X4v*l0P}^O;;fwQy!Q1H*ruV2>TeyeXRsp-csZI~RF?88`7xq;'
    '@H7lNTnlpp>OnRjnFp+%j7PS(;^Vr$aCS@;eLfrq$1>tMICdWA40;TH3Iq8a-i5xat+>!Kf@Bre;Y1^C`d<a-{g?Cm1z^$8(v%8C'
    'X_2HwJ@5NGB_+yMsFXxy7LtrenkpLFqN$Y7R6XzeJW7MMC@Mtp37Hv1;p_V^+^=)4>%M+DTaPKDS`HFXM=ovr*v9;+bAxiJFkGhD'
    'Ng@XI=zDtsZgGkQrpS4K=8PAR)1yGI=bb0T!xca-I-p2i2#QtKPy_kHczkCfsK+WWecgm-<$09oImR&W??1;ef%zb{evqy`I>^3S'
    'T0}fP{iP@F7m<?3QAFgSHgFw<0FJ6+|Jn1zYI7ou{4ksTl}V+$Z|dQzfB&7I@hP+-SPG{3IuoC_dq{$-CS<J;hXa>~NpYb+iUx_$'
    'w^cHDDbpMGp4Ox0u_AawKZE|-Yk+eHWw0s80)M=8g-<G8?BO9*5dCckhmV)zFP}6d_AgoCgEwK@+Vx<f+C)W`$dh-vO4aUat1xwI'
    'Fz&h{4u{J_@%fx^dikU&-R)OHxp7}1cv2!HOY<=<%#-*TtKs>BzNBHxJ+jK%fx_JdIC3eO=Z1pJz8!T~mMcuXy>{T=z4NHjB6GOn'
    'Ifd-qRD@<bBbenq=_C}^keYwoqniq$aFHYN`w>T7#TF1zn;qm4bB#Pymq*XLl(jy(0Mj17CC|VDk`1bfy_y<~)=0t9?s4QF-H%xN'
    'fb|mc0qFx85VcVsw2f8a?LI-GIa!@d@z(&6#58hh#aB9^GfsD!@23ZDZNM)g2kEA3*`(4(68n`Sh<~dDp0>MB-`W19K1;k|fpQ2F'
    '+&n>y-ICd(Qv1M?l_I<)kLjL<Yvi8e%YWVb77o5y4i4f)gme{=nje)^RmGb2oRR`{@eZt9!2y5QaypiL+-!97YbImuRU8?fO!|3u'
    '*?!MkH0$<l;?VDmP4<4|bE72OzED9+mYhI|Dn)Ypg&VG3778zYuM#RW1IDx@AjfM96`Z#hmny1*d)zlhu_%o4mh6NpUClHwV>V_w'
    '-(qA9obiuuGg+u`f)1^}K>E5>AmY(x$j~qXmAScQ;*Iy{@rj3^KO+tg=&Z$0G7O{t*AFaoWH3m;fe5dRp_=kRu)TH;_@9r&`*YGr'
    'eSHGNhHS^8=pG^!co$YJ&&Q;n>dcO64YI3Bn8>na?Bw6R@P0Ui#8C&VKUhq+tczx@O<6=664D{_&_5?^x<_{j>SICgW?U^3j+0Mb'
    'VQ#QXxDl)EK_+_*-MuoKePzifb_t41{FO?w#rFxxM=Pj#+DFskY&iACsi5L(ilt&NsmxNF>dw?VJcE4$bf=dH=HDzpPs!<IZ}1t2'
    '6;lR(eN*UG_}7z1^l^b*4k<C1f=Z2k^r?UZv%K&&%fuEiB?_NeeNkV=PEVJaIlqk2u1sevSI%X|B_|Wr`=6LO6ONo0T^~$Yp(e(~'
    'Q=aV4zsoA*i*l-pyx6}D^Ju!kLUwz!1zEv=$2&W_fyrWCvi1HYW|4}LR7tLo7`i6&L=<O|;8{u}Ogn^?IFQ0bZfavP6|2~5I=wto'
    '_dxdA9#yiWViUW0C$f*FHZqAEJNCaxnyhBPd-j3OQPv|wmB$>K#I}vdlGI)ySdrPwj9v8vXWJJ{YSt9gRlCaYO|HSW(+45LJPVs|'
    'm(U==FOdBw4r~@%QME&HXy7OdZ$?$o-S8ZZ(6oe3ja>Rt@E&2>Lty>qVKQ^^PPAIkz<#c_uWk)pNo`Y?!1nbEs=Un%_`BG8d|n3R'
    'ui`8A-qO=#YfTX^AXE=Bvagbv+XFE8>n?aW&4dVdSHMl#MierNhG%{v;2Sd^KfgBTTT9kc#bry-FxZpox_P0X#|_%?@*5+y?kf|Z'
    '_!UF?_k;P+89MSx0{$Fch!>xB;!}YkNZ;`TJlGsK*A)$!+w$3#pGpvGP6>ZJ3OhyPiQSJQ(A@eOVkSvLS8pcGEas84c7Mp-k^&9G'
    'XNZNmFqZD#PNU^4$Q{EJuul6)Pqw{gHibT=c5^QhV_6HF(;-Y(pRz&iFfJ4vDK~S$^AL9=hi<-DO1_+`py&EG<A~=N_!sWR?3Z;='
    'Q?vvY1p-L@E=9|?#;|7G2~YmYr_&~q$?<Qm*}%Sja=E7+Oyrcvyf2{yc}L0AmC@LB-xk-#1k?QLC=9qbN@lt}W|G?kVE$oS=6dik'
    'GCD1hdK8FreZ<~T37ck`{NOe&DtHZtHWFH_atwvzrJ?7+WaM8G!*_xQs9>7~oLjmYfBJqT^TO}p$(sjYr^sh){#65q9e!c5BZG}8'
    '-^j}!?$l3V8B}=RA;&BW@y2;E>=vE@GYr<l{M8N|o=Yy2H56c(col5?)r4Y!GMHzs20L}4{xMq#-4(kqUS=P>I`4t)8C7`wzaXqL'
    '{7vf9HbJEv!1?FbiKWRd_<M!n(hha*98M}#5YtC^6N&N?@#xu`0_pqQu%UyGH>c{dk@k!6luH1dFp&qPE4SE~oCcnCP93<Kv_tt;'
    'DZclOmCQZmZ!|u7gtlySCZTV#nIl)$g8Ah?ywH`JV4nI1s(tTql-4Fu!=6@{zbOR7q5xb^r$KE(IfP0|fR73fyUTAug~4m`9pmWA'
    '?_tC{{1ylqIndI)<uKJZ7w?JdV}n5wGkc<gFsF;j?uVQCGn041Gtc?V!C8~(LD>h8d2cNSN-I;Nrh{buu@L$$b|L;Z)1EdJtiqoB'
    '2(*5d29op?9eZU7zfa$R_hVMfvZHb+8*j(Vj<+N3CDMdvX$zN>L+EeY1(;msLu70SevCEY=hU`gy~iTB;jaj*e@TJOpJ5Q$I8Mbb'
    'Eu&fq2WZfoJ~CsG1dVV>1TQ^pTyr>`zHdZY6{HVtX;qk3mk!S~&f>9~4?upc5`-L-L&qygOxpz`!s|_jlyFDz|KbM;DL=`j$3o!c'
    'um|G5MZu5c*RYXWO4>UMIfvs!Vd3&T{1YsKhbOF{Ptb|PP4T5g$!+WnwHUY-A3`*#7FyLC;l37ivij!*y6?e$qI2DizVm#FxqEf!'
    'oQL+9TPDPBO)Dh=CS}l75{kutpAr3M{dCK+W*BZ0;M!R_puD*-dbhd|@bx6GgID7$(QsB#ECl+Ed2qWw9&Ep~<E)mwRLUS2hhy(j'
    'sq7heL@)<xtd7G;;bSm=o)*V@=rx#_1k*Fa1#p7NAzyCVfZNvN*tW40p6G`21cS$`WVI;r_<Atpr%g6wYhi7`R=T%%Etrj_u%zQ5'
    '$bB!y-1U?3V2c#?A4$fF2|>ok!5GH^$8hLmI8@np(C{6f$PWJ;y8nO(mRB`nN|6fpRZ=Iz5mDgF)JIT|G^FwNju1aP2#h9W(dA~('
    'NWjv6er${+8|TeL^-mG_;O0$cA|n%KnRe2UzdxwvvM{)DRu{CRSEH|6H+yfJ1m2U_L`web1;=CO==8j`D5t*_pYYSbu!MsypQW(F'
    '$rH08)ak`NQE2ON5f6L#&?h^dP>IjlkYIKd3sQV=Xr%?L7!Cq5Dh3UoR*|U<2RZ4hUCG?FDd=iGM3<*trr!E$G^foEr+L3%l;<YH'
    'A4zez6V(l^9cyTJVl?JGDZsQs8{k%7r-IRZ_QyX?I~kuQ<r)j9bHzo_sJnvoYEi^zu`w<TQsr+F5U0oNu49jwEBf5L3r>r@(bVK6'
    'l<`Ao-*Iz@un^|XIsJ{(Q<p*o{A2N_)-#sIjgoT#3Fxus6{Kb~L-3*-WaX))aPb?L>{+t|ALb>K9Y-%<+RPIC)%%3DD`tRW#YOhb'
    'pbstadPZti%D{o(Kh${IA^1^UkBv|7kSE#p<kxT`PRnovd4q-AP=l-V{lFE>RyE;&S3i$$62;(cXawl!L_*h-x13cb74)PLkBr?a'
    'geD7REId?4N91pkqFo}Cdc1_BC7~E8bOd<KXE5@{AYF2#lEOd%vn%5*dAd^ocG^nuH%B}s)|=N6&b$G-A(E0Q<QMsBe2=b^3WhqT'
    'LzETUPWDA-lD=_cu5pGnbzNn_NL0&#7B`nUx9<uJ!9yHusG-G*(p;U`Xx#jI6}HDd;8lJ=3?9oN+;SiKD5eRzJm-So+g9?EnFYJD'
    'GN3Yi5IaU+vmHE3G<bQ22_MrS-5$HKU0jqOy!sm)OQa~R{+EpPJHn0RkDPUF4fI!IC5CUyAa$-<Y~lDDhBm99eOVHW<&P6h6?^(b'
    '=nTq6KQ^m&jbJ@rJYqZ4KI4gw1c-Q$f_D3Raf<mioZOp<@1vaH*RgzD7p01)!!`r`WdQ&Enaj+!E(K|s9SHLk(d=>rdvkalF`jY>'
    '=CnEBOE+OKS{ILYRUzOSyc|jNMcnW~jt150!UdfV^vxkIG=?eiPwbI~9ivGo!94{NE5AYE>`1a~hdvv5bO9fa_tDZtcc@TLHSTLX'
    'jFuq5KiYBxoSAv(<ll@+HZ7QB`4$o^uY-EwDC@B>AC`5M(%%9~FnCt3dVJOiFj1b0lIh_nQFxbeHwlKI7sD{|U?$dowZTO*HsY2u'
    'ToAB&LVQ(~=+3zf#POgq1RN@5_DSA?pM_U2U-==LK6?(k-uQs3*9;tQ(S_?DF5xT}fE&HNoV4~pG(0dHRyQi}1z-+5kC;SdwlZLS'
    '*B@Vd`*SX@dyKv|@lgBo23=lo2}Vz@#aAhz=&hxJpOqq^xZ(-cZQBO7S~Y2elQPFB+y#A~P36D+DN82BJfsmmJ>=&)Us}^+4+eKD'
    'q3wDt*&lzLEj}_#+iwb?#`Wi*b!jOK58kK!&qHCvGXpLLN1^L@CKyVIp{=DNzy7x|Y}gzPYVwtI!74reiTwwGSs@HJe&(>**$dFr'
    '<v19GC!(|N9Wo_SfUo3K1B*7rV)I8^*tS>%?`p~6fY)BK`-KEuzao=_9n*q}{unqh91AHs#~`J48MZHOMro@gxJHHHrK28vR{w^-'
    'h4M%Tdxu=xcO6EYe6WPCNNc`JLY1XB)ZTat`|C|%Xk-<2zV{tZXo``ty>BsxJ;{caZbpFwU7A=f!S62&z>`x{sr~a81Qu?=jR($B'
    'dBF?R_wy}~>*__O?;0b%Mh0{b&w+wjrgVb=3u807n1I?5x?@fhV<#>M7X#%;S-UXjM_n%#3YY`he8ASM*_6Lc9RrqK!%^`wRK0nK'
    '&K|f#x{pSo=lM7EVecd0D6Yh7!M{oNsyApNwT^5M&St9bMl<%`BC){m7<ixXfuV*J<o*)Gp^5Vt<sJ<S3O(TY`ZFk*I7~EUk0F|N'
    'P|<_5+C{j7^FVZjY~&|HQO<d`Gp(KB9bE?T>wcqV%0&#@@QfrFoxtwJ=V^Iw9>j7DVg4QiSdc#*G(%&+$+wi;ntl?F`u+#mMjO#@'
    'bu9a>`x>Kv=nM&~{EvP1r~&WApT%#_258MX88qEj1NAOn>Hb~^d{A_YhW(kvy!JYbhGTiS{ZTFvI5a?ohcx(UtsjZDcP)rLZ9;kF'
    'X8gK(7PC%C8KzF&f{QkCv8pnZ2u5xo2Q+T7PQBB(n|}7w6LN1@;bmRS73$0?)h~hLn!?-yWjA7ICWvaGS>U9-5ghg_K$gQ^Dmf~M'
    'O=k1yKHYmn`DGW4)U9DpM-@=r$8p4Y&j_70kHdb?@}ss(S3`4xIff+$!mQb0czo&t=nI>Mp>>UPr1m)xG~SHbr9%ADs>RT3ML=-v'
    'ePVrM1|EybLe-*Z?9jXhX_xir#sA__X8U_OH0dk7B+~#vof#w~auq(z+C`d~AmVbohE}Ha5}wBeIvmGfO{Oimn$-z8CQIP{aaG`6'
    'TS#-ejd26V7DX=wlZD&5;D!7)?lbdd)>Q5#yG(R~j9lG-^OaSgw%|KSR?cE4H&kFq>LOGs9ieBA3=;2{c6jvbE^RrylJb+L@}KS2'
    'fS>!4ksBJuhKAmRL6cEfu>BT@{nCYT_8Y#kHvo~MK8Ptl1*6yQ(96rp$=#|q_|D7CZ1wY6@Z4(&byip4!nxyUP@ajxHW4HWG;!hO'
    'L|kUQA7s`>5tolPP|+#{T3hGhk|Gv4tLwq+!Cg9XPZzuuE9huI6z)DLLCqWHVypdA<bH1_sm_nVLg^FbCRl<;`6V{XO^d5M8iPqM'
    '%ZPE!BUm+MG6<_b<us%s*2yg<Zf#%58eya_*39K689v6H3tb^v)spgDlIc1S#l>;%aN+oFI3$@4ZBy1lh`2sHO5F<6&pZOB%snV^'
    'Sf9Hx_d2^QF%N{6PLMLYTTs(7MnwCQ$u0voI)0&!NS;xF1#g<Fi)|?QUj9qvTrMGR*Cg(R{$=oUT?T4=R3IxFMPcUG7w~zXEi|d6'
    'gGQ+tEFDbZEp(g=Q_~H(<$A|i_r#mDJYo^<Tl$eTuzm!%r2*f!@oDi|gr+e~HidTzw&+&Vz<0*zdan?4ExE9LWd`+-e+V@VdFWT('
    'fk(6)z-YECsXf|>bM8ptO%g*^n$M?JVaoiIJ(=XQ*&iC7q5*4u-^F1IB|`g!@#Lyjy5n6VIDPHFb<a&eYH<)VDYXFC>)6BD1!|~#'
    'po}dY>tQrDOhe82Vf10!ZT4?oBJ6qakh+$a6C;U2uzuOhoEve57q4GI%+-T*A%{cPO-h7|H!JX}yfAzl%qJ%Dt5HBUk~SK@CS_ZL'
    '2=~ids&Z~2c8MK?tfwlR;UkN|@WTdd=yrmm62^EZA)H3--i)Rib5M4j0(T?fF@eWrV4?DC{y@JT`cKtBDa|REIUI@0-#+3!mfML#'
    'Dn77OQVFk3y2-BpxSmFhM&a*}UmOXIYY-UIMC*$ra5Q9uajA2q8TTjiU7wPFyEzBP*9N0i**kV1?IB#wze4hdqKTmO71Sd0!To3h'
    '`=l?By^_hn{0-;vLEd$k6lB18;vtChByW@A+A3K6xtSPT-OL@<ttZ26?`XH&8Bh&zqdZdwGM0La>W@FcuR*&|;Nw@)Xwi#mCth-_'
    'W}d*LM^{0)JOi}{XG2r{7??%Z!-fxMaKH9!qV;GFh^DNAje&c><fjO%|I&m;#|<D{tPRg}J%*viX=L`c&2*wA5BT@Gz&^hfTyHM{'
    'PRlg*(N-^3rBW3<%%dS*^#vaOuNCjhe4@W*JSS29E5PVWWA*w~jYQp08@}uj0{H5VCkpdn^rHdTBp6}HW_Q{$b_RC6ybDWjo&{`5'
    'gT{_wqCfo|$s4l91?fUur-8LhpHm>|57EK!6((T$dlc-ZE8%ud4^fbx#z=>W;n1BgpuI8<yNyG@;L;~t_A3W2e2ayemm?V4o<ny^'
    '?*#tmtsq?ai!erE@aM^T>`Rd3KHi$f-dn6g_MNdnu|qskwuwVE4v$d5{SPsJ*;O9lMv>MHK^X8ok~l9cflpxoC62GbdXp00dd*#Y'
    '5#P?L)l>4}1D6pLFaVctCCq!fIM5NAV65uO$g35x=>G63^LxfSnCZHR&eoA9wrP2c)a^Z>d`29<mtUj0Z|>8CqE@Jq48&blhp?6t'
    'LqgV6LcIkAr+WureAa8~k!XuDH+gt0@dqQSLBIk8xgQoT1#SHk)dl)m+}HtiQhX?zE^3njXRF_g;xrrRbEsegVpE9D{c=2I5DJSd'
    'W`Mi;Gz<{Zgt%lr=rO;kn&&s7kUf|5J(J;j)_kVx-&ulzsXmsuy+?JE2W*7MMe@w1n7X`q$`%y`qQ<!(s&>R4f&{)$Yeioiv8;!q'
    '(-*<VfFYQ-ZH)drvK3;~9+9?%Q*mU=LFA1UlJqwfu*x9<M$#9N0C{Cn&#%E|*~9o}*9AQDa|X>hzYA8i7SqKmec60fBOHB}1ozt`'
    'QAR5rYy4+Iu;N})c)yok%l(Cm&#9nn%`&_}9I;++4tL0A7OG77Kw5sD1EZIs_+2ssGu8-kcmF8Di-8+h+m-34I6r}edbDwtB*=r1'
    '`)|DL7!8MH&XEs_-$?P75%P#Ol8>7%(SchN;KSKTtp<yMY&M35fN<P!bp@n98>6`@Vz5R00%W$N(J7PX5(~Wpupsq5$TUdsO;xr*'
    'p8f<)EQtoQ30LN}=6!H%T1vy(ro#-YOwf8ON&8<$v5_SVyUM*EI%dx2k|q}_{d^71ewqpMR`{ZNY#D6roJ(V(2Iz_1z04HT1UM$I'
    'fp6TC$zWtP3HFmF_a!x8(Tp-O@l$}YsY@cc$~T#ePjR%$rIgl4N7FalCR*6k$5zZ&;KSSt(9~do22)-!AL%;!bBzP&H)SI`B1Io}'
    'yx|$_F@y}0t?<`60-rZ0fzoac9lrmCy|vwr{VH++ad$k)D_5Z7uBlKcYY9Au2BNZ48}d^UAoI-`5Scg14hFfPzn(Py7rubAW0DB{'
    'G5soxZ12J=`WMIvO$D&sTZ0m{S4mjJWKe5i>6Sk&s4Zgv&PN0w@6vR>qh%$mezg#;Y8-$oO3JA8Ujw~hw-hA%0P?^8Vx}D2gI_KQ'
    'fkkf^oOdzD*iXk89(J%X#hPf~vX4~ns(_KbQP?=^KcZJYjK#JG@a2E|aLmLCKj|xzux>51>b$_oo4y;b6`6xh?PPAUaUF3y^N#(b'
    'bcZe0PbU5|DTsQ1gUkDZ*$b<@7>C@e_(Ss|ep(tp66MnI$r(BBpq&-=nu)Uw%kNWNeJ(iEETnM~(d6XnQe5>+64e|}V{u_4t(%?%'
    'rwjnSR;h!{Kp$K^=*CJ<bwQ02#&9m<Fbc#);(&1-9d{6BcUwtvPnXVv8UHx^IMtHr9QR`ueY}8gqI~~#X%7Z*!!i9uE4z1mCt8U8'
    'pfXmWaPfO8&9O=$@BNoUDnx=^YXuJa%!c6WF|c}DJEWFAN9jfTK}SIse78R%_n8P>;Mk0=EAOMjge3M23DM(ED3phJgP{8qzSdGl'
    'j8S|Aw(p8ik+%r;>b<9Nz2dahQUOYgC*iKYCd51aJXGHMz^sfP1iQc0G@`tSthgEhN3O|1?!_GFR2@Un_cu{tvmpe2amVzur5F&L'
    'hn(>#*d09?MDJ8W-R+Z@*mRH1++s@FvYh{QN+C=d<)L2dNBq0w2f0yR!&`bc2TEr5(s`>yLG<BVKIG)03MU+urr)RMF4xdal}S`i'
    'td(-ht}`o<kc^mMOsaE(!1ud|XjCUACsxxXQ90NpDgtIlZc#h;OU&(ut+ZnPepC(X#dSJuxNx^D92GSKUy)OIedjf@Q^(V+Vk8<4'
    'C)LqYZIf_(&NAYcvl!#nse#}AQnK@i8ZLMLO_kdhLSI!G{0ZL3-<&6g*PnXfK2=!^z$H{Q-v?(K?Z<Nm!qBmQ1DNT{vZtq{(<CKZ'
    'G<y37b}eruRYoCb+SQK1<GuK9dL8jq(1I3~`7qy95>K^sk|(DuP=@D$8&>~-F41F{+-ylZduBoS#0zS>s|Rq+Ez+5tLMLpaFv$UM'
    ';|DLatYYwHWi1rNL=wfIB``2;E!>XC!Y;W?UZ#Ny2#bs0$9@~yl#~ebd<#i1%_e@y=_K>54<=2&fl*@D$;{j!qBi3$YG0A&pZBmO'
    'vt2m6so_P;lZ)OsOG|_Mrz4H_)ocg<8!>SGQv`h5&_Z^bUqb21R+4ha8Hi^j%_zQ03?oJPF<YW=`j{k9<m+N$gCF5+l}Do>XUwy?'
    'f%Dl^c)F+@CHAZY3nvk7*}fU{T6_?ehnPXj%8$hSwLHprte`1X!ramX72G2(jWcq+ptJctEl3r>fx|nX&4`kQJH_af!>4ZFk6>+>'
    'CVulufWZD1%vA_NXU)SP{z)6-O}z2U#mQ(tDvIqdAJgxr9ig`RHJf7flR}p{cfQ{YzD#5mL`&_ZDqVc&ikgovcTeX&FNnj1cEu$6'
    '<Q)E|tOIbntQ^|Fo6gI>ijrxOs3c$w9lOL~>%3HaZej<ljs&;7%Nyn-F2R1EpP+v(g>8H^ng7l0Gtc>>E%JX|!lD;ea8<H_eX#W%'
    '*^^ib=7A36{W^Jm>y6ux4F=G#&lyS%UZxY-J#5uW9h6&agpJ%hq9ZSePdkn<B`1<ugCa2)Xj_aql?E_X@*AVIJ`CuHI1SU3;%iIE'
    '@ZCH!K*M%F#%s32pT&y6+v1Od7zy)YEI>$e10A#Vh6Q=b=y5#}9yxxYfgiVE&5=-Wat?ymVfx@zD+Q%Kq3Er47qY^9u+?@7+|CKW'
    '<QR9H<)qH9?Y3m%+k#L}Y7ZRfd5?Y<?ZLt<06e(%yxh7*Qr~!iNJU%1hhPcrpTCa?@pXXNB~l=k>p+D5p2hc>??K{LCUkvsMS+5^'
    'WHCL2+nSDG+slRcSELLjbB!@hISKN}D`F&c6=U1v;74i)c;!Wqt~>b<ksOAKX}>UZ&kYRJsie1Gp90MxcUIT_1qN2u;j&jIC_6Mv'
    'YoE-7qg&oV#_ke4uRTf(+-JhT<N>mOX(4V*GbRn)SJ7|aFSQ?x1aD^x(5pNRjV|%<Og9!ANg9xY0;v5m9<5yx@bcn)P$D6PuxAa%'
    'PY<Ui2R(>$^IA5-#S5ix+2OA3xA2aU2-VvwNw{XOnZdw#+$S6Wr!z8nS%W3)G{e1ERlXEe{<Z&(njo^(-+_9>ZYPFrmtnt|2!GFm'
    'Xl9@PQ95Y-j%}`yKvxV$0jX@dCsU7EH;+e5D`fD4qYAg^K^eX>4M!q13SrgHkb7q*{tdoF>8&aZ&R4|yzHd0^N08(Mn4%m{hTf``'
    '<a_?;LYqckP~YN%cY~yf!O5$*k#!;oz8`4V^+sYXdyaN*>nC1+yWsQBCiZ+BAK8b8@$G46+$GvfB_+>6OUYY$c$F^nfG%~Djr+Iv'
    '?VQKOGg13P6f|i)#PjVD7#|mg3taMPnyou7))>Vzso!a8xiB{)u!6JRG9F~6+yFx_KMa{T%2=zNXFF<pa7yhUy!e)iAGB^#tF<}k'
    'D3nd#F9>11ZY$#~`B>a~=L)^1&Sx*hKc}N&JXkLOn+O<|p|0>$^i>SR9a08pYSn~wLgH|4=pHtOe8jb`1H`$skAPl1(ThAuhR=Cn'
    '+&ekYef|^+#OKqPgVreiWCNMHV==wmrUe&GCUMuKH_<7Rim^ag0wi8F!Y^%q!rE4%i19<R>r@y0)uF=iocqta(p@z5%31W3a)8ds'
    '!R%eW5G(BakL&&Yba`wHC|lnqA|*HB$DPNR7Hy9sWqz2p$&oJIcoz<4yuq#Nd~lB0LluQ*a$PDuqxjx1EEf3>{aOv6Cx(kl${ZPI'
    '^*-i~qA@t6>T@FN<7mMR4t8iqz+&llY<OA#go?$Y-=zH@H19F$JUa`jZv}Wq?+jC^yma#6d=~Y)SP8dqGpZ)uB@g?8aAkEJFa^S-'
    '!bl1tz8*rWffsOoMl!izJA^7d@5#^HJRE%12v8x#%-xd<(jQXru%r)Mom7sh+bdcAs4=d(6$ghlcF@3{x!mcu&r#XmOQ=gsBIYh!'
    '#HPESW4f&f-mp0UFHW12S4gQd`Aqu-pX2AVdQ9i4R^q6<1VW!Jp?{kS;pkR-?y2j(sHppw`X|r9&w2*<XK4u4eap~u+VKz;HU&R@'
    'TFhVfUVwjR;x|v^U<;{}76x^*Y}j@<n0>9G4`lr{wshkdG3&HO(~n`;XQzNRa_6z)S1|Y`WzjMHvvg0@1Xg}6#iRNXSdujYl1F9X'
    '!lV{Rj(Ce%I;-jQoE{L_{+>*>(V=PQW}`BnrSYPtV9W9XY+Zd8e%HsK?~XDeVcJKHq?)PB)0r@BK{2uPOrq;Gm*V-s82IF`0P<&y'
    'S?Qo|eDiTC8h-i7MruW}{uA5iX$=KTtXz)ibR2&r#NtvV75HJK$hZb~!=J}Rc>M$yPsJ}I3E5S!E?%5lR*_2gDvR)EEa)L$eWmDw'
    '^HTgZ4@Ze_!(q5nb($)Tv1lb)hxS*b>A*Av>@O>U@k>r*Tg($kowop%dD%i+WhoZE7lXR9*P+OA5IS>oxZ_r-<idpsvLKaUztbMb'
    'OE^MJx5?AQvNU2Xun`7=M5*cCfA8rsneTIP8GN271{H{Ari_l$u0UJ7nVe6wy}cl%l>?4K_lbtS7K-Gxfs^DkRN1XUrKC&YKT%!!'
    '_t+CW`sx<gPfNmsu@}IeRmKMQ?Yz9bMO5lt3??54$3)#gu&Izokr7`!`}YEQv>_hvAN0YEH=9s@gaa1Km2e_J1eFY<z`$)k#;T~H'
    '>xbK<_V5-mW5-Xzoe@l@MqUQdOHNp4smIi~CqVd;$*A1(Zze=q@VGyp*ojNBZ_Z7F(rIZRI9nAr>Ai;cR~F+6Lx1?H@&Pvf|2Cj;'
    'DBk|4KzxL(aD!PBt<U@Chbu!AR%-JkOrL{m+XiSk;0inM#*vsHdrTQvhjD$4aPD~}cDe<imYpm~5x5NV95&&Wwhky6v?1lImf+`7'
    'ibmtPFi%kyRl;m=<Y)k%XWr7U?b*1aFAFB0O~<h?16Z>(5)|rH@fj0E#lpS;YaWeZk7LQE@gRJC>n0K8oc-5<SCMQTB1N9=oDKK-'
    'i1ao-{9vNlGhVjHK1l){e+j;g;s$y)<sv$yjMA_`C#-FBW4Z=yAg(eDJ4Fo9O(22RS;_KSSJ`9I>$C9wq7#h$HsA}qnh9e?M$q!n'
    '3%gf~<4lF$7(3@VXeZRO%enK&oxR<}FWVIUdP{*)z)V=V#tdcjLqTo&4D8v-!sRu3@b;Avthtnq9__lY?_(PsKWED?UwekUu((W~'
    'Dc?pR*UP9ia}w7_OqwgB{e=DaVmS$#_({G5?M7@j1UD^;ALnKeS=-<LoO6YoYVBjCs=DFspg65I--W$Tc(}3C1@`>w|3<cu@N>d('
    '>$pE%AkV?Rfs05~EphQ14)jzDLjsr+rl|z%<!yLx2l8-ZauRyiPojZ4YdAkvNzu+n)pXeRPxbx7VXVCB6*4hBkZpQ*k~&IlqFvkO'
    'GT(koXIq;0GnbBJah$AKW}xvpv%h67yQ%vU(LQ;I78vg*;(E4hjcqUcw(}VAwy!07pC&THwsE`%hrH>Pz$5JAY0lItx01c>=t;hd'
    '8PWlr$ZB^x4>oOk8uf7PW%VO>5DISejx3jySLcvZtL%u}DQWsiEQ0<G$fS$>S>o2w!fI?<L&FcXlRle3f=g^5&~FQIElQ?em#jgP'
    'Wd~rPg*6!7w}osI4Sf8c0Zt8=gTKnB!zHI6qysZxn});xZ~vp(P$%iK$tHu10{A1x9<S$WK>m(th>k~yQL_RpoU|4~9zQ47w=RSE'
    'j3IuOH-+}{QL=WP2rS>b0c=0*hpzq^_)5qe)W4p^uErbixqCMJPMn~7Arh}EspIFWC@eTBgG~8)?0pW<mEwRo8bWYT#tAo9|Dh@S'
    'OxXVcT*)IK'
)
# END EXPORTED TACTICAL VALUE WEIGHTS


def _decode_tactical_weights():
    values = iter(unpack("<6817f", decompress(b85decode(TACTICAL_WEIGHTS_B85))))

    def matrix(rows, columns):
        return tuple(
            tuple(next(values) for _column in range(columns))
            for _row in range(rows)
        )

    def vector(size):
        return tuple(next(values) for _index in range(size))

    weights = (
        matrix(48, TACTICAL_INPUTS),
        vector(48),
        matrix(48, 48),
        vector(48),
        matrix(1, 48),
        vector(1),
    )
    return weights


# Filled by the dependency-free exporter from the selected local checkpoint.
# Tuple unpacking happens once at module import; inference allocates no tensors.
TACTICAL_W1, TACTICAL_B1, TACTICAL_W2, TACTICAL_B2, TACTICAL_W3, TACTICAL_B3 = _decode_tactical_weights()

TUNE = {
    "gun_hunters": 1,        # SCOUTs sent to ram loaded enemy TANKs
    "transport_hunters": 2,   # SCOUTs assigned to deny high-value transports early
    "guard_cap": 2,          # SCOUTs that may peel off to kill a TRANSPORT's pursuer
    "guard_horizon": 7.0,    # seconds ahead a pursuit is treated as a real threat
    "guard_miss": 4.0,       # only guard against enemies already converging this close
    "keepers": 2,            # SCOUTs held back on our own goal mouth
    "keeper_post": 10.0,     # metres the keeper sits in front of our goal line
    "keeper_reach": 26.0,    # radius within which a keeper commits to a target
    "tank_station": 26.0,    # metres in front of our own goal line
    "tank_seek": 46.0,       # a TANK only holds station while something is this close
    "keeper_watch": 55.0,    # keepers stand down when no enemy is this near our goal
    "transport_fear": 2.4,   # enemy repulsion gain for TRANSPORTs
    "runner_fear": 0.0,      # SCOUTs on a scoring run do not dodge enemies: an even
                             # SCOUT-for-SCOUT trade removes a threat to a TRANSPORT
    "fear_radius": 7.0,
    "ram_radius": 3.0,       # opportunistic ram range for a running SCOUT
    "dodge_gain": 2.2,
    "dodge_trigger": 1.2,
    "dodge_lag": 0.22,       # seconds a target loses to control period and jerk
    "fire_floor": 0.3,       # expected points denied required to spend a round
    "shot_base": 0.35,       # worth of a round that the target can still dodge
    "patience": 26.0,        # seconds of slack before the magazine is dumped
    "gun_fear": 1.6,         # extra TRANSPORT repulsion from loaded enemy TANKs
    "gun_fear_range": 26.0,
    "block_cap": 2,          # SCOUTs allowed to body-block rounds for a TRANSPORT
    "block_range": 34.0,     # how far a loaded TANK counts as aiming at a ward
    "block_stand": 2.6,      # metres up the firing line the blocker sits
}

# Experiment 2 actor omitted: the accepted policy uses only the tactical value MLP.




MATCH_DURATION = 90.0          # the documented default; only used to pace gunnery


def _clamp(value, low, high):
    return low if value < low else high if value > high else value


def _dist(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < TINY:
        return _dist(point, start)
    t = _clamp(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq, 0.0, 1.0)
    return _dist(point, (start[0] + t * dx, start[1] + t * dy))


def _segment_hits_box(start, end, x_min, x_max, y_min, y_max):
    """Slab test: does the closed segment touch the axis-aligned box?"""
    enter, leave = 0.0, 1.0
    for origin, delta, low, high in (
        (start[0], end[0] - start[0], x_min, x_max),
        (start[1], end[1] - start[1], y_min, y_max),
    ):
        if -TINY < delta < TINY:
            if origin < low or origin > high:
                return False
            continue
        first, second = (low - origin) / delta, (high - origin) / delta
        if first > second:
            first, second = second, first
        if first > enter:
            enter = first
        if second < leave:
            leave = second
        if enter > leave:
            return False
    return True


def _closest_approach(rx, ry, rvx, rvy, horizon):
    """Smallest separation of two constant-velocity points over ``[0, horizon]``."""
    speed_sq = rvx * rvx + rvy * rvy
    if speed_sq < TINY:
        return hypot(rx, ry), 0.0
    when = _clamp(-(rx * rvx + ry * rvy) / speed_sq, 0.0, horizon)
    return hypot(rx + rvx * when, ry + rvy * when), when


def _pursuit_time(gap, target_velocity, offset, speed):
    """Time for a ``speed`` pursuer to reach a constant-velocity target."""
    vx, vy = target_velocity
    a = vx * vx + vy * vy - speed * speed
    b = 2.0 * (offset[0] * vx + offset[1] * vy)
    c = gap * gap
    if -TINY < a < TINY:
        return -c / b if b < -TINY else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = sqrt(disc)
    options = [value for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if value > 0.0]
    return min(options) if options else None


class SwarmController(BaseSwarmController):
    # ------------------------------------------------------------------ setup

    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.home = game_info.own_goal
        self.specs = dict(game_info.drone_specs)
        self.weapon = game_info.weapon_spec
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.forward = 1.0 if self.team is Team.A else -1.0
        self.duration = MATCH_DURATION
        self.goal_face = self.goal.x_min if self.team is Team.A else self.goal.x_max
        self.home_face = self.home.x_max if self.team is Team.A else self.home.x_min

        self._prepare_obstacles(game_info.obstacles)
        self._build_grid()
        self.attack = self._flow_field(self.goal)
        self.defend = self._flow_field(self.home)
        self._chain_cache = {}

        own = sorted(game_info.own_initial_drones, key=lambda drone: drone.id)
        opponent = sorted(game_info.opponent_initial_drones, key=lambda drone: drone.id)
        self._initial_counts = {
            self.team: {kind: sum(drone.drone_type is kind for drone in own) for kind in DroneType},
            self.team.opponent: {kind: sum(drone.drone_type is kind for drone in opponent) for kind in DroneType},
        }
        self._initial_value = sum(self.specs[drone.drone_type].point_value for drone in own)
        self.side = {drone.id: 1.0 if (drone.id % 2) == 0 else -1.0 for drone in own}
        self.lane = self._assign_lanes(own)
        self._last_command = {}
        self._tactical_assignments = {}
        self._tactical_overrides = {}
        self._next_tactical_time = 0.0

    def _assign_lanes(self, own):
        """Spread each class across the 14 m goal mouth so arrivals do not collide."""
        totals = {}
        for drone in own:
            totals[drone.drone_type] = totals.get(drone.drone_type, 0) + 1
        span = self.goal.y_max - self.goal.y_min - 2.0
        seen = {}
        lanes = {}
        for drone in own:
            rank = seen.get(drone.drone_type, 0)
            seen[drone.drone_type] = rank + 1
            lanes[drone.id] = self.goal.y_min + 1.0 + span * (rank + 0.5) / max(1, totals[drone.drone_type])
        return lanes

    def _prepare_obstacles(self, obstacles):
        self.circles = []
        self.boxes = []
        self.blobs = []          # (cx, cy, bounding radius, aabb) for coarse rejection
        for obstacle in obstacles:
            if isinstance(obstacle, CircleObstacle):
                cx, cy = obstacle.center
                radius = obstacle.radius
                self.circles.append((cx, cy, radius))
                self.blobs.append((cx, cy, radius, (cx - radius, cx + radius, cy - radius, cy + radius)))
            else:
                x0, x1 = obstacle.x_min, obstacle.x_max
                y0, y1 = obstacle.y_min, obstacle.y_max
                self.boxes.append((x0, x1, y0, y1))
                cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                self.blobs.append((cx, cy, hypot(x1 - x0, y1 - y0) * 0.5, (x0, x1, y0, y1)))

    def _blocked_point(self, x, y, clearance):
        for cx, cy, radius in self.circles:
            reach = radius + clearance
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy <= reach * reach:
                return True
        for x0, x1, y0, y1 in self.boxes:
            if x0 - clearance <= x <= x1 + clearance and y0 - clearance <= y <= y1 + clearance:
                return True
        return False

    # -------------------------------------------------------------- planning

    def _build_grid(self):
        self.nx = int(round(self.width / GRID)) + 1
        self.ny = int(round(self.height / GRID)) + 1
        blocked = bytearray(self.nx * self.ny)
        for _cx, _cy, _radius, (bx0, bx1, by0, by1) in self.blobs:
            i0 = max(0, int((bx0 - PLAN_CLEARANCE) / GRID) - 1)
            i1 = min(self.nx - 1, int((bx1 + PLAN_CLEARANCE) / GRID) + 1)
            j0 = max(0, int((by0 - PLAN_CLEARANCE) / GRID) - 1)
            j1 = min(self.ny - 1, int((by1 + PLAN_CLEARANCE) / GRID) + 1)
            for j in range(j0, j1 + 1):
                row = j * self.nx
                y = j * GRID
                for i in range(i0, i1 + 1):
                    if not blocked[row + i] and self._blocked_point(i * GRID, y, PLAN_CLEARANCE):
                        blocked[row + i] = 1
        self.blocked = blocked
        self.clearance = self._chamfer(blocked)
        self.weight = [
            1.0 + COMFORT_WEIGHT * (COMFORT - value) / COMFORT if value < COMFORT else 1.0
            for value in self.clearance
        ]

    def _chamfer(self, blocked):
        """Two-pass approximate distance (metres) from each cell to blocked space."""
        big = 1.0e6
        nx, ny = self.nx, self.ny
        field = [0.0 if flag else big for flag in blocked]
        for j in range(ny):
            row = j * nx
            below = row - nx
            for i in range(nx):
                index = row + i
                value = field[index]
                if value == 0.0:
                    continue
                if i > 0 and field[index - 1] + 1.0 < value:
                    value = field[index - 1] + 1.0
                if j > 0:
                    if field[below + i] + 1.0 < value:
                        value = field[below + i] + 1.0
                    if i > 0 and field[below + i - 1] + SQ2 < value:
                        value = field[below + i - 1] + SQ2
                    if i + 1 < nx and field[below + i + 1] + SQ2 < value:
                        value = field[below + i + 1] + SQ2
                field[index] = value
        for j in range(ny - 1, -1, -1):
            row = j * nx
            above = row + nx
            for i in range(nx - 1, -1, -1):
                index = row + i
                value = field[index]
                if value == 0.0:
                    continue
                if i + 1 < nx and field[index + 1] + 1.0 < value:
                    value = field[index + 1] + 1.0
                if j + 1 < ny:
                    if field[above + i] + 1.0 < value:
                        value = field[above + i] + 1.0
                    if i > 0 and field[above + i - 1] + SQ2 < value:
                        value = field[above + i - 1] + SQ2
                    if i + 1 < nx and field[above + i + 1] + SQ2 < value:
                        value = field[above + i + 1] + SQ2
                field[index] = value
        return [value * GRID for value in field]

    def _flow_field(self, zone):
        """Dijkstra cost-to-go towards ``zone`` plus successor pointers."""
        nx, ny = self.nx, self.ny
        cost = [float("inf")] * (nx * ny)
        nxt = [-1] * (nx * ny)
        heap = []
        i0 = max(0, int(zone.x_min / GRID))
        i1 = min(nx - 1, int(round(zone.x_max / GRID)))
        j0 = max(0, int(zone.y_min / GRID) + 1)
        j1 = min(ny - 1, int(zone.y_max / GRID) - 1)
        for j in range(j0, j1 + 1):
            row = j * nx
            for i in range(i0, i1 + 1):
                index = row + i
                if not self.blocked[index]:
                    cost[index] = 0.0
                    heappush(heap, (0.0, index))
        weight = self.weight
        blocked = self.blocked
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))
        while heap:
            here, index = heappop(heap)
            if here > cost[index] + TINY:
                continue
            j, i = divmod(index, nx)
            for di, dj, length in steps:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                target = nj * nx + ni
                if blocked[target]:
                    continue
                value = here + length * GRID * weight[target]
                if value + TINY < cost[target]:
                    cost[target] = value
                    nxt[target] = index
                    heappush(heap, (value, target))
        return cost, nxt

    def _cell(self, position):
        i = _clamp(int(position[0] / GRID + 0.5), 0, self.nx - 1)
        j = _clamp(int(position[1] / GRID + 0.5), 0, self.ny - 1)
        return j * self.nx + i

    def _open_cell(self, position, cost):
        """Nearest cell that is both free and connected to the destination."""
        index = self._cell(position)
        if not self.blocked[index] and cost[index] != float("inf"):
            return index
        nx = self.nx
        j0, i0 = divmod(index, nx)
        for radius in range(1, 12):
            best, best_gap = -1, 1.0e9
            for dj in range(-radius, radius + 1):
                nj = j0 + dj
                if nj < 0 or nj >= self.ny:
                    continue
                span = range(-radius, radius + 1) if abs(dj) == radius else (-radius, radius)
                for di in span:
                    ni = i0 + di
                    if ni < 0 or ni >= nx:
                        continue
                    candidate = nj * nx + ni
                    if self.blocked[candidate] or cost[candidate] == float("inf"):
                        continue
                    gap = di * di + dj * dj
                    if gap < best_gap:
                        best, best_gap = candidate, gap
            if best >= 0:
                return best
        return index

    def _cost_to_go(self, field, position):
        cost = field[0]
        return cost[self._open_cell(position, cost)]

    def _chain(self, field_id, nxt, index):
        """Cached ladder of candidate waypoints along the flow path from a cell.

        Only the handful of points the line-of-sight probe actually tries are
        kept, which keeps the cache small enough to be irrelevant next to the
        sandbox memory limit however long the match runs.
        """
        key = (field_id, index)
        cached = self._chain_cache.get(key)
        if cached is not None:
            return cached
        points = []
        node = index
        for _ in range(CHAIN):
            step = nxt[node]
            if step < 0:
                break
            node = step
            j, i = divmod(node, self.nx)
            points.append((i * GRID, j * GRID))
        last = len(points) - 1
        ladder, seen = [], set()
        for offset in (last, last * 3 // 4, last // 2, last // 3, last // 5, last // 8, 2, 1, 0):
            if 0 <= offset <= last and offset not in seen:
                seen.add(offset)
                ladder.append(points[offset])
        chain = tuple(ladder)
        if len(self._chain_cache) > 30000:
            self._chain_cache.clear()
        self._chain_cache[key] = chain
        return chain

    def _visible(self, start, end, margin=LOS_MARGIN):
        lo_x, hi_x = (start[0], end[0]) if start[0] <= end[0] else (end[0], start[0])
        lo_y, hi_y = (start[1], end[1]) if start[1] <= end[1] else (end[1], start[1])
        for cx, cy, radius in self.circles:
            reach = radius + margin
            if cx + reach < lo_x or cx - reach > hi_x or cy + reach < lo_y or cy - reach > hi_y:
                continue
            if _point_segment_distance((cx, cy), start, end) <= reach:
                return False
        for x0, x1, y0, y1 in self.boxes:
            if x1 + margin < lo_x or x0 - margin > hi_x or y1 + margin < lo_y or y0 - margin > hi_y:
                continue
            if _segment_hits_box(start, end, x0 - margin, x1 + margin, y0 - margin, y1 + margin):
                return False
        return True

    def _waypoint(self, position, field_id):
        """Furthest visible point along the flow-field path from ``position``."""
        cost, nxt = self.attack if field_id == 0 else self.defend
        chain = self._chain(field_id, nxt, self._open_cell(position, cost))
        if not chain:
            return None
        for point in chain:
            if self._visible(position, point):
                return point
        return chain[-1]

    def _route(self, drone, spec, destination, field_id=None):
        """Steer straight at ``destination`` when visible, otherwise via the field."""
        if self._visible(drone.position, destination):
            return self._track(drone, destination, spec)
        if field_id is not None:
            waypoint = self._waypoint(drone.position, field_id)
            if waypoint is not None:
                return self._track(drone, waypoint, spec)
        return self._track(drone, destination, spec)

    # -------------------------------------------------------------- steering

    def _track(self, drone, target, spec, arrive=None):
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        remaining = hypot(dx, dy)
        if remaining < TINY:
            desired = (0.0, 0.0)
        else:
            speed = spec.max_speed
            if arrive is not None:
                speed = min(speed, sqrt(2.0 * spec.max_acceleration * max(0.0, remaining - arrive)))
            desired = (dx / remaining * speed, dy / remaining * speed)
        return (2.6 * (desired[0] - drone.velocity[0]), 2.6 * (desired[1] - drone.velocity[1]))

    def _near_obstacles(self, position, reach):
        """Obstacles whose inflated bounds sit within ``reach`` of ``position``."""
        x, y = position
        circles = [item for item in self.circles
                   if abs(item[0] - x) <= reach + item[2] and abs(item[1] - y) <= reach + item[2]]
        boxes = [item for item in self.boxes
                 if item[0] - reach <= x <= item[1] + reach and item[2] - reach <= y <= item[3] + reach]
        return circles, boxes

    def _crashes(self, drone, spec, acceleration):
        """Replay the engine's jerk-limited dynamics and test the swept path."""
        px, py = drone.position
        vx, vy = drone.velocity
        cax, cay = drone.acceleration
        dax, day = acceleration
        magnitude = hypot(dax, day)
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            dax, day = dax * scale, day * scale
        reach = hypot(vx, vy) * HORIZON + 0.5 * spec.max_acceleration * HORIZON * HORIZON + 1.0
        circles, boxes = self._near_obstacles((px, py), reach)
        if not circles and not boxes:
            return False
        jerk = spec.max_jerk * SUB_DT
        margin = 0.25 + CRASH_MARGIN
        for _ in range(int(HORIZON / SUB_DT)):
            jx, jy = dax - cax, day - cay
            span = hypot(jx, jy)
            if span > jerk:
                scale = jerk / span
                jx, jy = jx * scale, jy * scale
            cax, cay = cax + jx, cay + jy
            ex = px + vx * SUB_DT + 0.5 * cax * SUB_DT * SUB_DT
            ey = py + vy * SUB_DT + 0.5 * cay * SUB_DT * SUB_DT
            vx, vy = vx + cax * SUB_DT, vy + cay * SUB_DT
            speed = hypot(vx, vy)
            if speed > spec.max_speed:
                scale = spec.max_speed / speed
                vx, vy = vx * scale, vy * scale
            for cx, cy, radius in circles:
                if _point_segment_distance((cx, cy), (px, py), (ex, ey)) <= radius + margin:
                    return True
            for x0, x1, y0, y1 in boxes:
                if _segment_hits_box((px, py), (ex, ey), x0 - margin, x1 + margin, y0 - margin, y1 + margin):
                    return True
            px, py = ex, ey
        return False

    def _safe_acceleration(self, drone, spec, acceleration):
        if not self._crashes(drone, spec, acceleration):
            return acceleration
        ax, ay = acceleration
        base = hypot(ax, ay)
        if base < TINY:
            ax, ay, base = self.forward, 0.0, 1.0
        ux, uy = ax / base, ay / base
        limit = spec.max_acceleration
        preferred = self.side.get(drone.id, 1.0)
        for angle in FAN:
            for sign in (preferred, -preferred):
                turn = angle * sign
                cos_t, sin_t = cos(turn), sin(turn)
                candidate = ((ux * cos_t - uy * sin_t) * limit, (ux * sin_t + uy * cos_t) * limit)
                if not self._crashes(drone, spec, candidate):
                    return candidate
        speed = hypot(drone.velocity[0], drone.velocity[1])
        if speed > TINY:
            return (-drone.velocity[0] / speed * limit, -drone.velocity[1] / speed * limit)
        return (0.0, 0.0)

    # ------------------------------------------------------------------ step

    def step(self, state):
        try:
            return self._decide(state)
        except Exception:
            return {
                drone.id: self._last_command.get(drone.id, (self.forward, 0.0))
                for drone in state.own_drones
                if drone.status is DroneStatus.ACTIVE
            }

    def _decide(self, state):
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        foes = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        duties = self._plan(state, own, foes)
        actions = {}
        for drone in own:
            spec = self.specs[drone.drone_type]
            role, mark = duties[drone.id]
            if role is GUN:
                acceleration = self._tank_move(drone, spec, foes)
            elif role is HUNT and mark is not None:
                acceleration = self._chase(drone, spec, mark)
            elif role is BLOCK and mark is not None:
                acceleration = self._block(drone, spec, mark)
            elif role is KEEP:
                acceleration = self._keep(drone, spec, foes)
            else:
                acceleration = self._goal_run(drone, spec, foes)
            acceleration = self._avoid_enemies(drone, spec, foes, role, mark, acceleration)
            acceleration = self._avoid_friends(drone, spec, own, acceleration)
            acceleration = self._dodge(drone, spec, state, acceleration)
            acceleration = self._safe_acceleration(drone, spec, acceleration)
            self._last_command[drone.id] = acceleration
            if drone.drone_type is DroneType.TANK:
                aim = self._gunnery(drone, own, foes, state)
                if aim is not None:
                    actions[drone.id] = {"acceleration": acceleration, "fire_direction": aim}
                    continue
            actions[drone.id] = acceleration
        return actions

    # ------------------------------------------------------------------ plan

    def _policy_features(self, state, own, foes):
        """Fixed-size pooled features, independent of live swarm size."""
        own_by_type = {kind: [] for kind in DroneType}
        foe_by_type = {kind: [] for kind in DroneType}
        for drone in own:
            own_by_type[drone.drone_type].append(drone)
        for drone in foes:
            foe_by_type[drone.drone_type].append(drone)

        own_transports = own_by_type[DroneType.TRANSPORT]
        foe_transports = foe_by_type[DroneType.TRANSPORT]

        def mean_progress(drones, field):
            if not drones:
                return 1.0
            return sum(1.0 - _clamp(self._cost_to_go(field, d.position) / self.width, 0.0, 1.0) for d in drones) / len(drones)

        def min_ttg(drones, field):
            if not drones:
                return 0.0
            return _clamp(min(self._cost_to_go(field, d.position) / max(self.specs[d.drone_type].max_speed, TINY) for d in drones) / self.duration, 0.0, 1.0)

        def mean_forward_speed(drones, direction):
            if not drones:
                return 0.0
            return sum(_clamp(direction * d.velocity[0] / max(self.specs[d.drone_type].max_speed, TINY), -1.0, 1.0) for d in drones) / len(drones)

        def survival(drones, team, kind):
            return len(drones[kind]) / max(1, self._initial_counts[team][kind])

        enemy_ammo = sum((d.shots_remaining or 0) for d in foe_by_type[DroneType.TANK])
        own_ammo = sum((d.shots_remaining or 0) for d in own_by_type[DroneType.TANK])
        pressure = sum(self._cost_to_go(self.defend, d.position) < 30.0 for d in foes)
        threatened = 0
        loaded_guns = [d for d in foe_by_type[DroneType.TANK] if d.shots_remaining]
        loaded_own = [d for d in own_by_type[DroneType.TANK] if d.shots_remaining]
        for ward in own_transports:
            if any(_dist(ward.position, gun.position) < TUNE["block_range"] for gun in loaded_guns):
                threatened += 1
        pursued = sum(
            any(_dist(ward.position, scout.position) < 12.0 for scout in foe_by_type[DroneType.SCOUT])
            for ward in own_transports
        )
        enemy_projectiles = sum(shot.team is self.team.opponent for shot in state.projectiles)

        return (
            _clamp((self.duration - state.time) / self.duration, 0.0, 1.0),
            _clamp((state.own_score - state.opponent_score) / 40.0, -1.0, 1.0),
            len(own_by_type[DroneType.SCOUT]) / 14.0,
            len(own_transports) / 8.0,
            len(own_by_type[DroneType.TANK]) / 4.0,
            len(foe_by_type[DroneType.SCOUT]) / 14.0,
            len(foe_transports) / 8.0,
            len(foe_by_type[DroneType.TANK]) / 4.0,
            mean_progress(own_transports, self.attack),
            mean_progress(foe_transports, self.defend),
            min_ttg(own_transports, self.attack),
            min_ttg(foe_transports, self.defend),
            enemy_ammo / 20.0,
            own_ammo / 20.0,
            pressure / 26.0,
            threatened / max(1, len(own_transports)),
            survival(own_by_type, self.team, DroneType.SCOUT),
            survival(own_by_type, self.team, DroneType.TRANSPORT),
            survival(own_by_type, self.team, DroneType.TANK),
            survival(foe_by_type, self.team.opponent, DroneType.SCOUT),
            survival(foe_by_type, self.team.opponent, DroneType.TRANSPORT),
            survival(foe_by_type, self.team.opponent, DroneType.TANK),
            state.own_score / max(1, self._initial_value),
            state.opponent_score / max(1, self._initial_value),
            mean_forward_speed(own_transports, self.forward),
            mean_forward_speed(foe_transports, -self.forward),
            mean_progress(own_by_type[DroneType.SCOUT], self.attack),
            mean_progress(foe_by_type[DroneType.SCOUT], self.defend),
            len(loaded_guns) / max(1, self._initial_counts[self.team.opponent][DroneType.TANK]),
            len(loaded_own) / max(1, self._initial_counts[self.team][DroneType.TANK]),
            enemy_projectiles / 20.0,
            pursued / max(1, len(own_transports)),
        )

    def _entity_features(self, drone, friendly):
        kind = drone.drone_type
        spec = self.specs[kind]
        field = self.attack if friendly else self.defend
        cost = self._cost_to_go(field, drone.position)
        speed = hypot(*drone.velocity)
        max_speed = max(spec.max_speed, TINY)
        max_acceleration = max(spec.max_acceleration, TINY)
        return (
            float(kind is DroneType.SCOUT),
            float(kind is DroneType.TRANSPORT),
            float(kind is DroneType.TANK),
            _clamp(self.forward * (drone.position[0] - self.width / 2.0) / (self.width / 2.0), -1.0, 1.0),
            _clamp((drone.position[1] - self.height / 2.0) / (self.height / 2.0), -1.0, 1.0),
            _clamp(self.forward * drone.velocity[0] / max_speed, -1.0, 1.0),
            _clamp(drone.velocity[1] / max_speed, -1.0, 1.0),
            _clamp(self.forward * drone.acceleration[0] / max_acceleration, -1.0, 1.0),
            _clamp(drone.acceleration[1] / max_acceleration, -1.0, 1.0),
            1.0 - _clamp(cost / self.width, 0.0, 1.0),
            (drone.shots_remaining or 0) / 5.0,
            POINT_VALUE[kind] / 5.0,
            _clamp(speed / max_speed, 0.0, 1.5),
            _clamp(cost / max_speed / self.duration, 0.0, 1.0),
        )

    def _pair_features(self, scout, role, candidate):
        _key, target, ward = candidate
        target_entity = self._entity_features(target, False)
        dx = self.forward * (target.position[0] - scout.position[0]) / self.width
        dy = (target.position[1] - scout.position[1]) / self.height
        scout_speed = max(self.specs[DroneType.SCOUT].max_speed, TINY)
        dvx = self.forward * (target.velocity[0] - scout.velocity[0]) / scout_speed
        dvy = (target.velocity[1] - scout.velocity[1]) / scout_speed
        distance = _dist(target.position, scout.position)
        if ward is None:
            ward_dx = ward_dy = ward_progress = 0.0
        else:
            ward_dx = self.forward * (ward.position[0] - scout.position[0]) / self.width
            ward_dy = (ward.position[1] - scout.position[1]) / self.height
            ward_progress = 1.0 - _clamp(self._cost_to_go(self.attack, ward.position) / self.width, 0.0, 1.0)
        return tuple(float(index == role) for index in range(6)) + target_entity + (
            _clamp(dx, -1.0, 1.0),
            _clamp(dy, -1.0, 1.0),
            _clamp(dvx, -2.0, 2.0),
            _clamp(dvy, -2.0, 2.0),
            _clamp(distance / self.width, 0.0, 1.0),
            _clamp(distance / scout_speed / self.duration, 0.0, 1.0),
            target_entity[-1],
            POINT_VALUE[target.drone_type] / 5.0,
            (target.shots_remaining or 0) / 5.0,
            _clamp(ward_dx, -1.0, 1.0),
            _clamp(ward_dy, -1.0, 1.0),
            ward_progress,
        )

    def _tactical_score(self, features):
        values = tuple(
            tanh(bias + sum(weight * value for weight, value in zip(row, features)))
            for row, bias in zip(TACTICAL_W1, TACTICAL_B1)
        )
        values = tuple(
            tanh(bias + sum(weight * value for weight, value in zip(row, values)))
            for row, bias in zip(TACTICAL_W2, TACTICAL_B2)
        )
        return TACTICAL_B3[0] + sum(
            weight * value for weight, value in zip(TACTICAL_W3[0], values)
        )

    def _tactical_context(self, state, own, foes, scouts):
        wards = [drone for drone in own if drone.drone_type is DroneType.TRANSPORT]
        transports = sorted(
            (drone for drone in foes if drone.drone_type is DroneType.TRANSPORT),
            key=lambda drone: drone.id,
        )
        tanks = sorted(
            (drone for drone in foes if drone.drone_type is DroneType.TANK and drone.shots_remaining),
            key=lambda drone: drone.id,
        )
        guard_candidates = []
        for target, _catch_time in self._pursuers(wards, foes, scouts):
            ward = min(wards, key=lambda item: (_dist(item.position, target.position), item.id))
            guard_candidates.append((("DRONE", int(target.id)), target, ward))
        block_candidates = [
            (("BLOCK_LINE", int(ward.id), int(gun.id)), gun, ward)
            for ward, gun in self._gun_lines(wards, foes)
        ]

        def maximum_progress(drones, field):
            return max(
                (1.0 - _clamp(self._cost_to_go(field, drone.position) / self.width, 0.0, 1.0) for drone in drones),
                default=1.0,
            )

        def nearest_distance(first, second):
            if not first or not second:
                return 1.0
            return _clamp(min(_dist(left.position, right.position) for left in first for right in second) / self.width, 0.0, 1.0)

        foe_scouts = [drone for drone in foes if drone.drone_type is DroneType.SCOUT]
        loaded_tanks = [drone for drone in foes if drone.drone_type is DroneType.TANK and drone.shots_remaining]
        scout_count = max(1, len(scouts))
        global_features = tuple(self._policy_features(state, own, foes)) + (
            maximum_progress(wards, self.attack),
            maximum_progress(transports, self.defend),
            nearest_distance(wards, foe_scouts),
            nearest_distance(wards, loaded_tanks),
            _clamp(len(guard_candidates) / scout_count, 0.0, 1.0),
            _clamp(len(block_candidates) / scout_count, 0.0, 1.0),
        )
        shared = {
            HUNT_TRANSPORT: [(('DRONE', int(target.id)), target, None) for target in transports],
            HUNT_TANK: [(('DRONE', int(target.id)), target, None) for target in tanks],
            GUARD_TRANSPORT: guard_candidates,
            TACTICAL_BLOCK: block_candidates,
        }
        return global_features, shared

    def _tactical_options(self, state, scout, global_features, shared, keep_possible):
        previous_role, _previous_target, since = self._tactical_assignments.get(
            int(scout.id), (TACTICAL_RUN, None, state.time)
        )
        duration = _clamp((state.time - since) / 10.0, 0.0, 1.0)
        scout_features = self._entity_features(scout, True)
        previous = tuple(float(index == previous_role) for index in range(6))
        options = [(TACTICAL_RUN, None, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0) + (0.0,) * 26)]
        if keep_possible:
            options.append((TACTICAL_KEEP, None, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0) + (0.0,) * 26))
        for role in (HUNT_TRANSPORT, HUNT_TANK, GUARD_TRANSPORT, TACTICAL_BLOCK):
            ranked = sorted(
                ((self._pair_features(scout, role, candidate), candidate) for candidate in shared.get(role, ())),
                key=lambda item: (item[0][25], repr(item[1][0])),
            )
            if ranked:
                pair_features, candidate = ranked[0]
                options.append((role, candidate, pair_features))
        scored = []
        for role, candidate, pair_features in options:
            features = global_features + scout_features + pair_features + previous + (duration,)
            scored.append((self._tactical_score(features), role, candidate))
        return scored

    def _update_tactical_assignments(self, state, own, foes):
        scouts = sorted(
            (drone for drone in own if drone.drone_type is DroneType.SCOUT),
            key=lambda drone: drone.id,
        )
        global_features, shared = self._tactical_context(state, own, foes, scouts)
        live_scout_ids = {int(scout.id) for scout in scouts}
        locked = any(
            scout_id in live_scout_ids and state.time + TINY < override[2]
            for scout_id, override in self._tactical_overrides.items()
        )
        proposal = None
        if not locked:
            keep_possible = self._goal_threatened(foes)
            for scout_index, scout in enumerate(scouts):
                scored = self._tactical_options(state, scout, global_features, shared, keep_possible)
                logits = [option[0] for option in scored]
                peak = max(logits)
                probabilities = [exp(logit - peak) for logit in logits]
                best_index = max(range(len(scored)), key=lambda index: (logits[index], -index))
                confidence = probabilities[best_index] / sum(probabilities)
                _score, role, candidate = scored[best_index]
                if role == TACTICAL_RUN or confidence < TACTICAL_CONFIDENCE:
                    continue
                candidate_proposal = (confidence, -scout_index, scout, role, candidate)
                if proposal is None or candidate_proposal[:2] > proposal[:2]:
                    proposal = candidate_proposal

        selected = {}
        if proposal is not None:
            _confidence, _order, scout, role, candidate = proposal
            target_key = candidate[0] if candidate is not None else None
            selected[int(scout.id)] = (role, target_key)
            self._tactical_overrides[int(scout.id)] = (
                role,
                target_key,
                state.time + TACTICAL_COMMITMENT,
            )

        updated = {}
        for scout in scouts:
            role, target_key = selected.get(int(scout.id), (TACTICAL_RUN, None))
            old_role, old_target, old_since = self._tactical_assignments.get(
                int(scout.id), (TACTICAL_RUN, None, state.time)
            )
            since = old_since if (old_role, old_target) == (role, target_key) else state.time
            updated[int(scout.id)] = (role, target_key, since)

        live_keys = {candidate[0] for candidates in shared.values() for candidate in candidates}
        active = {}
        for scout_id, (role, target_key, until) in self._tactical_overrides.items():
            if scout_id not in updated or state.time + TINY >= until:
                continue
            if target_key is not None and target_key not in live_keys:
                continue
            old_role, old_target, old_since = updated[scout_id]
            since = old_since if (old_role, old_target) == (role, target_key) else state.time
            updated[scout_id] = (role, target_key, since)
            active[scout_id] = (role, target_key, until)
        self._tactical_assignments = updated
        self._tactical_overrides = active

    def _resolve_tactical_duties(self, state, own, foes):
        own_by_id = {int(drone.id): drone for drone in own}
        foes_by_id = {int(drone.id): drone for drone in foes}
        duties = {}
        for drone in own:
            if drone.drone_type is DroneType.TANK:
                duties[drone.id] = (GUN if drone.shots_remaining else RUN, None)
            elif drone.drone_type is DroneType.TRANSPORT:
                duties[drone.id] = (RUN, None)
            else:
                role, target_key, _since = self._tactical_assignments.get(
                    int(drone.id), (TACTICAL_RUN, None, state.time)
                )
                if role == TACTICAL_KEEP:
                    duties[drone.id] = (KEEP, None)
                elif role == TACTICAL_BLOCK and target_key and target_key[0] == "BLOCK_LINE":
                    ward = own_by_id.get(int(target_key[1]))
                    tank = foes_by_id.get(int(target_key[2]))
                    duties[drone.id] = (BLOCK, (ward, tank)) if ward is not None and tank is not None else (RUN, None)
                elif target_key and target_key[0] == "DRONE":
                    target = foes_by_id.get(int(target_key[1]))
                    duties[drone.id] = (HUNT, target) if target is not None else (RUN, None)
                else:
                    duties[drone.id] = (RUN, None)
        return duties

    def _plan(self, state, own, foes):
        """Run the tiny learned value policy; execute duties deterministically."""
        if state.time + TINY >= self._next_tactical_time:
            self._update_tactical_assignments(state, own, foes)
            self._next_tactical_time = state.time + TACTICAL_INTERVAL
        return self._resolve_tactical_duties(state, own, foes)

    def _gun_targets(self, foes):
        """Loaded enemy TANKs, nearest to our own goal first.

        A TANK is only worth one point as a body, but the five rounds it is
        holding are worth several TRANSPORTs.  It is also the slowest thing on
        the board, so a SCOUT that spends itself on one is a bargain.
        """
        guns = [foe for foe in foes if foe.drone_type is DroneType.TANK and foe.shots_remaining]
        guns.sort(key=lambda gun: (-(gun.shots_remaining or 0), self._cost_to_go(self.defend, gun.position)))
        return guns

    def _pursuers(self, wards, foes, free):
        """Enemies that will reach one of our TRANSPORTs before we reach the goal.

        A TRANSPORT cannot outrun a SCOUT, so fleeing only delays the trade; the
        answer is to spend a SCOUT of our own on the pursuer, which turns a
        five-for-one loss into an even one.
        """
        if not wards or not free:
            return []
        horizon = TUNE["guard_horizon"]
        scout_speed = self.specs[DroneType.SCOUT].max_speed
        threats = {}
        for ward in wards:
            for foe in foes:
                offset = (ward.position[0] - foe.position[0], ward.position[1] - foe.position[1])
                gap = hypot(offset[0], offset[1])
                if gap > horizon * self.specs[foe.drone_type].max_speed:
                    continue
                catch = _pursuit_time(gap, ward.velocity, offset, self.specs[foe.drone_type].max_speed)
                if catch is None or catch > horizon:
                    continue
                miss, _when = _closest_approach(
                    -offset[0], -offset[1],
                    foe.velocity[0] - ward.velocity[0], foe.velocity[1] - ward.velocity[1],
                    horizon,
                )
                if miss > TUNE["guard_miss"]:
                    continue
                reach = min(
                    (_pursuit_time(
                        _dist(scout.position, foe.position),
                        foe.velocity,
                        (foe.position[0] - scout.position[0], foe.position[1] - scout.position[1]),
                        scout_speed,
                    ) or 1.0e6)
                    for scout in free
                )
                if reach > catch + 1.0:
                    continue
                if catch < threats.get(foe.id, (None, 1.0e6))[1]:
                    threats[foe.id] = (foe, catch)
        return sorted(threats.values(), key=lambda item: item[1])

    def _gun_lines(self, wards, foes):
        """(ward, tank) pairs where a loaded enemy TANK has a clear firing line."""
        guns = [foe for foe in foes if foe.drone_type is DroneType.TANK and foe.shots_remaining]
        if not guns:
            return []
        reach = TUNE["block_range"]
        lines = []
        for ward in wards:
            best, best_gap = None, reach
            for gun in guns:
                gap = _dist(gun.position, ward.position)
                if gap < best_gap and self._visible(gun.position, ward.position, margin=0.0):
                    best, best_gap = gun, gap
            if best is not None:
                lines.append((ward, best))
        return lines

    def _goal_threatened(self, foes):
        """Is any enemy still close enough to our goal to be worth guarding against?"""
        watch = TUNE["keeper_watch"]
        return any(self._cost_to_go(self.defend, foe.position) < watch for foe in foes)

    def _goal_run(self, drone, spec, foes):
        if abs(drone.position[0] - self.goal_face) < 14.0:
            lane = _clamp(self.lane.get(drone.id, self.goal.center[1]), self.goal.y_min + 1.0, self.goal.y_max - 1.0)
            target = (self.goal_face + self.forward * 2.0, lane)
        else:
            target = self._waypoint(drone.position, 0)
            if target is None:
                target = (self.goal_face + self.forward * 2.0, self.goal.center[1])
        acceleration = self._track(drone, target, spec)
        if drone.drone_type is DroneType.SCOUT:
            ram = self._free_kill(drone, foes)
            if ram is not None:
                return self._track(drone, ram, spec)
        return acceleration

    def _free_kill(self, drone, foes):
        """A high-value enemy close enough that ramming barely costs progress."""
        reach = TUNE["ram_radius"]
        best, best_gap = None, reach
        for foe in foes:
            if foe.drone_type is not DroneType.TRANSPORT:
                continue
            gap = _dist(drone.position, foe.position)
            if gap < best_gap:
                best, best_gap = foe, gap
        if best is None:
            return None
        return (best.position[0] + best.velocity[0] * 0.25, best.position[1] + best.velocity[1] * 0.25)

    def _chase(self, drone, spec, mark):
        offset = (mark.position[0] - drone.position[0], mark.position[1] - drone.position[1])
        gap = hypot(offset[0], offset[1])
        when = _pursuit_time(gap, mark.velocity, offset, spec.max_speed)
        if when is None:
            when = gap / max(spec.max_speed, TINY)
        when = min(when, 4.0)
        aim = (mark.position[0] + mark.velocity[0] * when, mark.position[1] + mark.velocity[1] * when)
        return self._route(drone, spec, aim, field_id=1)

    def _keep(self, drone, spec, foes):
        """Guard the mouth of our own goal and body-block anything that arrives."""
        mouth = (self.home_face + self.forward * TUNE["keeper_post"],
                 _clamp(self.lane.get(drone.id, self.home.center[1]), self.home.y_min + 1.0, self.home.y_max - 1.0))
        best, best_cost = None, None
        reach = TUNE["keeper_reach"]
        for foe in foes:
            if _dist(foe.position, mouth) > reach:
                continue
            cost = self._cost_to_go(self.defend, foe.position) - 6.0 * self.specs[foe.drone_type].point_value
            if best_cost is None or cost < best_cost:
                best, best_cost = foe, cost
        if best is not None:
            return self._chase(drone, spec, best)
        return self._route(drone, spec, mouth, field_id=1)

    def _block(self, drone, spec, mark):
        """Stand on the firing line: a round stopped by a SCOUT costs one point."""
        ward, gun = mark
        dx = gun.position[0] - ward.position[0]
        dy = gun.position[1] - ward.position[1]
        span = hypot(dx, dy)
        if span < TINY:
            return self._goal_run(drone, spec, ())
        stand = TUNE["block_stand"]
        post = (ward.position[0] + dx / span * stand, ward.position[1] + dy / span * stand)
        return self._route(drone, spec, post, field_id=0)

    def _tank_move(self, drone, spec, foes):
        """Hold a firing station while there is anything to shoot, else advance.

        A TANK that sits on an empty station for the whole match wastes both its
        magazine and the point its body is worth, which is exactly what happens
        once its half of the arena has been cleared.
        """
        reach = TUNE["tank_seek"]
        if not any(_dist(drone.position, foe.position) < reach for foe in foes):
            return self._goal_run(drone, spec, foes)
        station_x = self.home_face + self.forward * TUNE["tank_station"]
        if self.forward * (drone.position[0] - station_x) < 0.0:
            station = (station_x, _clamp(self.lane.get(drone.id, self.height * 0.5), 4.0, self.height - 4.0))
            if self._visible(drone.position, station):
                return self._track(drone, station, spec, arrive=0.4)
            waypoint = self._waypoint(drone.position, 0)
            return self._track(drone, waypoint or station, spec)
        return self._track(drone, drone.position, spec)

    # ------------------------------------------------------------ reflexes

    def _avoid_enemies(self, drone, spec, foes, role, mark, acceleration):
        if drone.drone_type is DroneType.TRANSPORT:
            gain = TUNE["transport_fear"]
        elif role in (HUNT, KEEP, BLOCK):
            gain = 0.0
        else:
            gain = TUNE["runner_fear"]
        if gain <= 0.0:
            return acceleration
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        reach = TUNE["fear_radius"]
        spared = getattr(mark, "id", None)
        for foe in foes:
            if foe.id == spared:
                continue
            rx, ry = px - foe.position[0], py - foe.position[1]
            if rx * rx + ry * ry > reach * reach * 2.25:
                continue
            rvx, rvy = vx - foe.velocity[0], vy - foe.velocity[1]
            miss, when = _closest_approach(rx, ry, rvx, rvy, 2.5)
            if miss >= reach * 0.5:
                continue
            mx, my = rx + rvx * when, ry + rvy * when
            span = hypot(mx, my)
            if span < TINY:
                mx, my, span = rx, ry, max(hypot(rx, ry), TINY)
            push = spec.max_acceleration * gain * (1.0 - miss / (reach * 0.5))
            ax += mx / span * push
            ay += my / span * push
        if drone.drone_type is DroneType.TRANSPORT:
            ax, ay = self._keep_off_guns(drone, spec, foes, ax, ay)
        return ax, ay

    def _keep_off_guns(self, drone, spec, foes, ax, ay):
        """A TRANSPORT inside a loaded TANK's short range cannot dodge; back off."""
        danger = TUNE["gun_fear_range"]
        px, py = drone.position
        for foe in foes:
            if foe.drone_type is not DroneType.TANK or not foe.shots_remaining:
                continue
            rx, ry = px - foe.position[0], py - foe.position[1]
            gap = hypot(rx, ry)
            if gap > danger or gap < TINY:
                continue
            if not self._visible(foe.position, drone.position, margin=0.0):
                continue
            push = spec.max_acceleration * TUNE["gun_fear"] * (1.0 - gap / danger)
            ax += rx / gap * push
            ay += ry / gap * push
        return ax, ay

    def _avoid_friends(self, drone, spec, own, acceleration):
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        for friend in own:
            if friend.id == drone.id:
                continue
            rx, ry = px - friend.position[0], py - friend.position[1]
            if rx * rx + ry * ry > 49.0:
                continue
            rvx, rvy = vx - friend.velocity[0], vy - friend.velocity[1]
            miss, when = _closest_approach(rx, ry, rvx, rvy, 1.4)
            if miss >= 2.2:
                continue
            mx, my = rx + rvx * when, ry + rvy * when
            span = hypot(mx, my)
            if span < TINY:
                mx, my, span = rx, ry, hypot(rx, ry)
                if span < TINY:
                    mx, my, span = 1.0, 0.0, 1.0
            push = spec.max_acceleration * 1.7 * (2.2 - miss) / 2.2
            ax += mx / span * push
            ay += my / span * push
        return ax, ay

    def _dodge(self, drone, spec, state, acceleration):
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        trigger = TUNE["dodge_trigger"]
        for shot in state.projectiles:
            if shot.source_drone_id == drone.id:
                continue
            rx, ry = shot.position[0] - px, shot.position[1] - py
            if rx * rx + ry * ry > 1600.0:
                continue
            rvx, rvy = shot.velocity[0] - vx, shot.velocity[1] - vy
            miss, when = _closest_approach(rx, ry, rvx, rvy, 3.0)
            if miss >= trigger or when <= TINY:
                continue
            span = hypot(rvx, rvy)
            if span < TINY:
                continue
            cross = rx * rvy - ry * rvx
            side = self.side.get(drone.id, 1.0) if abs(cross) < TINY else (1.0 if cross > 0 else -1.0)
            push = spec.max_acceleration * TUNE["dodge_gain"] * (1.0 - miss / trigger + 0.35)
            ax += -rvy / span * side * push
            ay += rvx / span * side * push
        return ax, ay

    # --------------------------------------------------------------- gunnery

    def _lead(self, origin, target):
        speed = self.weapon.projectile_speed
        rx = target.position[0] - origin[0]
        ry = target.position[1] - origin[1]
        vx, vy = target.velocity
        a = vx * vx + vy * vy - speed * speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        times = []
        if -TINY < a < TINY:
            if abs(b) > TINY:
                times.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = sqrt(disc)
                times.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        flight = min((value for value in times if value > 0.0), default=0.0)
        return (target.position[0] + vx * flight, target.position[1] + vy * flight), flight

    def _hit_chance(self, foe, flight):
        """How much lateral room the target has to leave the 0.75 m contact disc.

        A vehicle only starts evading once it can see the round, loses about a
        fifth of a second to the control period and its own jerk limit, and is
        then bounded by its class acceleration.  TRANSPORTs are so sluggish that
        anything under roughly a second of flight is effectively unavoidable,
        which is exactly the shot worth waiting for.
        """
        spec = self.specs[foe.drone_type]
        lag = max(0.0, flight - TUNE["dodge_lag"])
        escape = 0.5 * spec.max_acceleration * lag * lag
        certain = _clamp(1.0 - 0.92 * escape / 0.75, 0.0, 1.0)
        # Even a round that is dodged costs the target its heading, so a shot is
        # never worth zero; ``shot_base`` is that floor.
        base = TUNE["shot_base"]
        return base + (1.0 - base) * certain

    def _gunnery(self, tank, own, foes, state):
        if not tank.shots_remaining or tank.next_fire_time is None or state.time + TINY < tank.next_fire_time:
            return None
        speed = self.weapon.projectile_speed
        floor = self._fire_floor(tank, state)
        best, best_score = None, floor
        for foe in foes:
            aim, flight = self._lead(tank.position, foe)
            if flight <= 0.0 or flight > 2.6:
                continue
            span = _dist(tank.position, aim)
            if span < TINY:
                continue
            if not self._visible(tank.position, aim, margin=0.0):
                continue
            dirx = (aim[0] - tank.position[0]) / span
            diry = (aim[1] - tank.position[1]) / span
            if self._friendly_in_line(tank, own, dirx * speed, diry * speed, flight):
                continue
            score = self._line_value(tank, foes, dirx * speed, diry * speed)
            if score > best_score:
                best, best_score = (dirx, diry), score
        return best

    def _line_value(self, tank, foes, pvx, pvy, span=2.6):
        """Expected points denied by one round, counting everything on the line."""
        total = 0.0
        for foe in foes:
            rx = tank.position[0] - foe.position[0]
            ry = tank.position[1] - foe.position[1]
            miss, when = _closest_approach(rx, ry, pvx - foe.velocity[0], pvy - foe.velocity[1], span)
            if miss >= 0.75 or when <= TINY:
                continue
            total += self.specs[foe.drone_type].point_value * self._hit_chance(foe, when)
        return total

    def _fire_floor(self, tank, state):
        """Be choosy while there is time to wait, then dump the magazine."""
        left = max(0.0, self.duration - state.time)
        needed = tank.shots_remaining * self.weapon.cooldown
        slack = left - needed
        if slack > TUNE["patience"]:
            return TUNE["fire_floor"]
        if slack > TUNE["patience"] * 0.4:
            return TUNE["fire_floor"] * 0.4
        return 0.05

    def _friendly_in_line(self, tank, own, pvx, pvy, flight):
        for friend in own:
            if friend.id == tank.id:
                continue
            rx = tank.position[0] - friend.position[0]
            ry = tank.position[1] - friend.position[1]
            miss, _ = _closest_approach(rx, ry, pvx - friend.velocity[0], pvy - friend.velocity[1], flight)
            if miss < 0.95:
                return True
        return False
