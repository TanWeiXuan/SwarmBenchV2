# Security model

Controller submissions are arbitrary untrusted Python. Source scanning catches mistakes but is not a security boundary.

## Official execution

Untrusted PR validation and tournament compute jobs explicitly receive only `contents: read`. They have no secrets, repository/discussion/PR write permission, or privileged token. Controllers run as persistent unprivileged Docker processes with:

- `--network none`;
- read-only container filesystem and controller bind mount;
- a 128 MiB `noexec,nosuid` tmpfs;
- one CPU, 2 GiB memory, and 128-process limits;
- bounded wall-clock watchdogs and captured logs;
- a scrubbed/dedicated RNG and CPU-thread environment.

Trusted reporter/publisher jobs never run `automation compute`, import submission files, or check out an unmerged PR head. They consume only versioned JSON validated for exact schedule identity, primitive types, numeric ranges, expected path, and head SHA. Artifact strings are not evaluated as code. `pull_request_target` is used only after merge for trusted current-state publication and checks out `main`.

Ratings update only after every batch validates. Discussion failure reporting and official current-state publication are separate from controller compute. Workflow permissions are declared at job level.

## Local execution

The default local backend is only process isolation for development; it is not hostile-code containment. A local controller can access your account and files with your permissions. Inspect it or use a disposable container/VM. Never run an unknown submission on a workstation containing credentials.

Deterministic execution is best effort for arbitrary third-party native code. Official runners use CPU inference and thread limits, but the trusted engine's determinism does not make hostile or native controller behavior safe or portable.

