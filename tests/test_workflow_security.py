from pathlib import Path


ROOT = Path(__file__).parents[1]


def workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_untrusted_submission_workflow_is_read_only() -> None:
    text = workflow("submission-validation.yml")
    assert "pull_request_target" not in text
    assert "permissions:\n  contents: read" in text
    assert "pull-requests: write" not in text
    assert "discussions: write" not in text
    assert "SWARMBENCH_BACKEND: docker" in text


def test_privileged_reporter_never_checks_out_or_runs_controller_code() -> None:
    text = workflow("submission-reporter.yml")
    workflow_permissions = text.split("jobs:", 1)[0]
    assert "pull_request_target:" not in text
    assert "contents: read" in workflow_permissions
    assert "actions: read" in workflow_permissions
    assert "pull-requests: read" in workflow_permissions
    assert "contents: write" not in workflow_permissions
    assert "actions: write" not in workflow_permissions
    assert "pull-requests: write" not in workflow_permissions
    assert "actions/checkout" not in text
    assert "python" not in text.lower()
    assert "submission.py" not in text
    assert "markPullRequestReadyForReview" in text
    assert "github.rest.pulls.merge" in text


def test_privileged_reporter_merges_with_github_app_token() -> None:
    text = workflow("submission-reporter.yml")
    read_jobs = text.index("- name: Read validation jobs")
    app_token = text.index("- name: Create short-lived submission token")
    update = text.index("- name: Update sticky progress and automatically merge")
    app_token_step = text[app_token:text.index("- name: Download final validated result")]
    update_step = text[update:]
    assert "actions/create-github-app-token@v3" in text
    assert "app-id: ${{ vars.SWARMBENCH_APP_ID }}" in text
    assert "private-key: ${{ secrets.SWARMBENCH_APP_PRIVATE_KEY }}" in text
    assert read_jobs < app_token < update
    assert "github.rest.actions.listJobsForWorkflowRun" in text[read_jobs:app_token]
    assert "permission-actions" not in app_token_step
    assert "permission-contents: write" in app_token_step
    assert "permission-pull-requests: write" in app_token_step
    assert "github-token: ${{ steps.app-token.outputs.token }}" in update_step
    assert "github.rest.actions.listJobsForWorkflowRun" not in update_step
    assert "VALIDATION_JOBS: ${{ steps.validation-jobs.outputs.jobs }}" in update_step
    assert "JSON.parse(process.env.VALIDATION_JOBS)" in update_step
    assert "enablePullRequestAutoMerge" not in update_step
    assert "pull.head.sha !== run.head_sha" in update_step
    assert "sha: run.head_sha" in update_step
    assert "createWorkflowDispatch" not in text
    assert "submission-accepted.yml" not in text


def test_privileged_reporter_resolves_live_fork_pull_requests_by_validated_head() -> None:
    text = workflow("submission-reporter.yml")
    job_header = text.split("  report:", 1)[1].split("    steps:", 1)[0]
    assert "workflow_run.pull_requests" not in job_header
    assert "listPullRequestsAssociatedWithCommit" not in text
    assert "github.rest.pulls.list" in text
    assert "state: 'open'" in text
    assert "base: context.payload.repository.default_branch" in text
    assert "head: `${headOwner}:${run.head_branch}`" in text
    assert "candidate.head.sha === run.head_sha" in text
    assert "candidate.head.repo?.full_name === run.head_repository.full_name" in text


def test_privileged_reporter_downloads_results_only_for_submission_prs() -> None:
    text = workflow("submission-reporter.yml")
    resolve = text.index("- name: Resolve validated submission PR")
    download = text.index("- name: Download final validated result")
    update = text.index("- name: Update sticky progress and automatically merge")
    download_step = text[download:update]
    assert resolve < download < update
    assert "core.setOutput('pr_number', prNumber)" in text[resolve:download]
    assert "steps.submission.outputs.pr_number" in download_step
    assert "continue-on-error" not in download_step
    assert "if: ${{ steps.submission.outputs.pr_number }}" in text[update:]
    assert "const prNumber = Number(process.env.PR_NUMBER);" in text[update:]


def test_acceptance_workflow_checks_out_only_merged_main() -> None:
    text = workflow("submission-accepted.yml")
    assert "pull_request_target" in text
    assert "workflow_dispatch:" in text
    assert "actions: read" in text
    assert "ref: main" in text
    assert "ref: ${{ github.event.pull_request.head" not in text
    assert 'pulls/$PR_NUMBER/files' not in text
    assert "--jq '.merge_commit_sha'" in text
    assert "git diff-tree --no-commit-id --name-only" in text
    assert '"$merge_sha^" "$merge_sha" -- submissions' in text
    assert "${#submission_paths[@]} -ne 1" in text
    assert "competition.publisher" in text
    assert 'pip install -e ".[competition]"' in text


def test_controller_dockerfile_drops_root() -> None:
    text = (ROOT / "Dockerfile.controller").read_text(encoding="utf-8")
    assert "COPY requirements-controller.txt pyproject.toml README.md ./" in text
    assert "USER 65534:65534" in text


def test_tournament_jobs_install_automation_dependencies() -> None:
    installs = [line for line in workflow("tournament.yml").splitlines() if "pip install -e" in line]
    assert installs
    assert all('".[competition]"' in line for line in installs)


def test_bot_publishers_use_github_app_token() -> None:
    for name in ("submission-accepted.yml", "tournament.yml"):
        text = workflow(name)
        assert "actions/create-github-app-token@v3" in text
        assert "app-id: ${{ vars.SWARMBENCH_APP_ID }}" in text
        assert "private-key: ${{ secrets.SWARMBENCH_APP_PRIVATE_KEY }}" in text
        assert "token: ${{ steps.app-token.outputs.token }}" in text
        assert "GH_TOKEN: ${{ steps.app-token.outputs.token }}" in text
        assert "gh workflow run tests.yml" not in text
        assert "gh workflow run submission-validation.yml" not in text


def test_rating_publishers_share_lock_until_their_pr_merges() -> None:
    tournament_final = workflow("tournament.yml").split("  final:", 1)[1]
    accepted = workflow("submission-accepted.yml").split("  initialize-rating:", 1)[1]
    for publisher in (tournament_final, accepted):
        assert "group: swarmbench-rating-publication" in publisher
        assert "cancel-in-progress: false" in publisher
        assert 'gh pr merge "$url" --auto --squash' in publisher
        assert 'bash .github/scripts/wait-for-pr-merge.sh "$url"' in publisher

    waiter = (ROOT / ".github" / "scripts" / "wait-for-pr-merge.sh").read_text(encoding="utf-8")
    assert "mergeStateStatus" in waiter
    assert "MERGED)" in waiter
    assert '[[ "$merge_state" == "DIRTY" ]]' in waiter
    assert "Timed out waiting" in waiter


def test_submission_jobs_install_validation_dependencies() -> None:
    installs = [line for line in workflow("submission-validation.yml").splitlines() if "pip install -e" in line]
    assert installs
    assert all('".[competition]"' in line for line in installs)


def test_tournament_compute_and_report_permissions_are_separated() -> None:
    text = workflow("tournament.yml")
    assert 'cron: "17 */6 * * *"' in text
    assert text.count("SWARMBENCH_BACKEND: docker") == 1
    assert text.count("Maintain live tournament Discussion") == 1
    assert "compute_matrix: ${{ steps.compute_matrix.outputs.value }}" in text

    compute = text.split("  compute:", 1)[1].split("  final:", 1)[0]
    assert "needs: prepare" in compute
    assert "fail-fast: false" in compute
    assert "max-parallel: 19" in compute
    assert "matrix: ${{ fromJSON(needs.prepare.outputs.compute_matrix) }}" in compute
    assert "contents: read" in compute
    assert "discussions: write" not in compute
    assert "pull-requests: write" not in compute

    reporter = text.split("  reporter:", 1)[1].split("  compute:", 1)[0]
    assert "contents: read" in reporter
    assert "actions: read" in reporter
    assert "discussions: write" in reporter
    assert "automation live-report" in reporter
    assert "automation compute" not in reporter

    final = text.split("  final:", 1)[1]
    assert "needs: compute" in final
    assert "contents: read" in final
    assert "actions: read" in final
    assert "discussions: write" not in final
    assert "automation final" in final
    assert "automation compute" not in final
