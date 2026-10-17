# Autonomous task execution and background jobs

Mark-LV's autonomous task system extends the existing `core.task_engine`; it is
not a second planner. The `task_agent` action queues a goal immediately; a
worker thread creates a bounded plan from the currently registered actions and
executes eligible steps through the normal `core.action_loader` dispatch used
by the app. It does not replace the Gemini Live conversation, voice, avatar,
UI, plugin loader, or system prompt.

## Normal use

For a goal that needs several actions, tell JARVIS what you want in natural
language. The `task_agent` action queues the goal; the worker builds and persists
a bounded plan, then executes eligible steps in the background without asking
for one command per step. Mark-LV reports a job ID; use the `jobs` action to
check status, inspect the report or logs, cancel, or resume the job:

- `list` or `history` — recent jobs and their states.
- `status` / `report` with a `job_id` — plan steps, results, blockers and recent
  log entries. `status` without an ID lists recent jobs.
- `logs` with a `job_id` — recent progress records.
- `cancel` with an ID — stop after the active action returns; queued/waiting jobs
  are cancelled immediately.
- `resume` with an ID — resume a waiting or queued job.
- `submit` with a `goal` — enqueue a background goal explicitly.

A step outside the autonomous profile is put in `waiting`, not silently
skipped. A step gets the configured number of argument-repair retries; after
those are exhausted, the worker can request at most `max_replans` alternative
plans. Earlier completed steps are retained, and the failed/unfinished suffix
is preserved in plan history. If argument repair changes a step into an
operation that needs approval, the repaired arguments are persisted and that
step waits for confirmation rather than being retried automatically. For steps
that are available through the existing confirmation flow, use `task_agent`
with `action=run` and that plan's ID to
approve/run the step, then resume the background job. If approval is not
available or the action is not registered, the blocker is reported in the job;
nothing is fabricated.

## Enable and configure

Background jobs are enabled by the default profile when no local configuration
exists. To customize the profile, copy `config/autonomy.example.json` to
`config/autonomy.json` and edit it. The local file is git-ignored because it may
encode a user's security policy. It is read at application start and is not
hot-reloaded; restart Mark-LV after changing it. These switches govern automatic
background execution; they do not remove the user's ability to use an existing
manual action/confirmation flow.

Important settings:

| Setting | Default | Meaning |
|---|---:|---|
| `allow_autonomous` | `true` | Master switch for automatic execution; false makes background steps wait. |
| `enable_background_jobs` | `true` | Enables the worker and persistent job queue. |
| `autonomous_tools` | small low-impact allow-list | Only these registered actions may run automatically; unknown tools fail closed. |
| `max_job_runtime_seconds` | `300` | Maximum continuous worker time per run before the job waits for resume. |
| `max_concurrent_jobs` | `2` | Maximum concurrently executing jobs. |
| `max_queued_jobs` | `100` | Maximum non-terminal jobs retained in the active queue. |
| `max_plan_steps` | `6` | Maximum steps per plan. |
| `max_retries` | `1` | Retries after the initial step attempt. |
| `max_replans` | `1` | Bounded alternative plan after a step exhausts retries; set to `0` to disable. |
| `retry_backoff_seconds` | `1.0` | Bounded pause between attempts (0–60 seconds). |
| `allow_network_actions` | `true` | Allows network-using actions in the allow-list; false requires manual confirmation for web/search/research and broad file/code actions that may send content to the configured model provider. This is not an OS firewall and does not disable the planner or provider calls used for planning/repair. |
| `allow_file_writes` | `false` | Writes do not run automatically; manual confirmation remains available through existing gates. |
| `allow_destructive` | `false` | Destructive operations require approval unless deliberately enabled. |
| `allow_external_facing` | `false` | Sending/publishing or acting for the user externally requires approval. |
| `respect_os_permissions` | `true` | Never elevates privileges; checks current-directory writability where possible. The target action still reports target-specific permission failures. |

Risky tools do not become autonomous just because they exist or a model selected
them. To deliberately change policy, edit `autonomous_tools` and the relevant
permission switches together. Unknown capabilities default to confirmation.
Security-sensitive tools remain confirmation-gated. The action still runs as
the current OS user; Mark-LV does not acquire administrator/root privileges.

The default profile allows `think`, `web_search`, `research`,
`file_processor` (only an explicit read-only action with `save: false`), and
`code_helper` (only `action: explain`). It also retains operation-aware
permissions from Mark-LV's existing action policy: `file_controller` reads and
searches, known `open_app` aliases, and reversible volume/brightness/dark-mode
adjustments may run routinely. File creation, folder creation, and append
require `allow_file_writes: true`; overwrites, deletes, moves, copies, renames,
and unknown app launches stay behind confirmation. High-impact computer
controls (closing apps/windows, locking the screen, changing WiFi, shutdown, or
restart) always require explicit confirmation in task plans, even if the generic
destructive-action profile setting is changed. `weather_report` and
`flight_finder` launch a browser and therefore wait for
confirmation; `flight_finder` may also save a report to Desktop. Use the
read-only `web_search` action for a headless weather lookup. Every planned
action must also be present in the live action registry; `think` is the task
engine's internal model-only step. Plugin tools are not currently planned or
executed by the task worker. Research stores its brief in Mark-LV's existing
local memory, and AI file/code analysis sends the selected content through the
configured model provider. Turning `allow_network_actions` off puts those broad
file/code actions behind confirmation; it is not a network sandbox. Use a
local provider or OS firewall if outbound model traffic must be prevented
entirely.

## Persistence, restart recovery, and shutdown

- `memory/jobs.json` stores job ID, goal, state, timestamps, attempt count,
  plan ID and snapshot, results/errors, cancellation flag, and recent logs.
- The existing `memory/tasks.json` stores canonical plans, ordered steps,
  per-step states/attempts/results/errors, and the retry policy for the plan.
- Both JSON stores use a lock and write to a temporary file followed by atomic
  replacement. If state JSON is corrupt, it is preserved as `*.corrupt` before
  the app starts a new store; it is not silently overwritten.
- The worker starts once during the real `JarvisLive.run()` lifecycle, not once
  per Live reconnect. At startup, queued jobs are dispatched and interrupted
  `running` jobs are recovered. Successful steps are not rerun.
- If a process crashes while a known read-only autonomous tool is in flight,
  that step is marked for a bounded retry. If an operation may have changed
  state, the job waits for human review rather than replaying it automatically;
  a reviewer sees a duplicate-action warning before explicitly authorizing a
  retry.
- On normal UI shutdown or the assistant's shutdown command, the worker gets a
  short grace period to save at a step boundary. Work not finished is left
  queued/running for recovery at the next launch. A currently blocked external
  call cannot be forcibly killed safely.
- Jobs can make progress only while the application, computer, required model
  provider, network and integrations are available. Nothing runs while the
  computer is powered off. After restart, work that needs a provider or
  credential resumes only when that dependency is available; failure details
  are recorded and shown rather than reported as success.

## Dependencies and truthful failure reporting

Planning and model-based recovery use the configured one-shot provider ladder
(`core.gemini`), while voice conversation remains on the original Gemini Live
path. Network actions require connectivity. Individual integrations can require
API keys, OAuth tokens, installed OS tools, browser binaries, or a logged-in
session. The task runner calls actual registered actions and their normal error
handling; it cannot grant permissions, create credentials, or infer that an
empty/failed response means success. Missing credentials, offline providers,
unavailable tools, retry exhaustion, and runtime limits appear as a failure or
blocker in job status and reports.

## Known limitations

- JSON persistence is appropriate for this single-user desktop app; it is not a
  multi-process/distributed queue and is not a guarantee against disk failure.
- A process crash during an in-flight read-only operation can cause that read to
  be repeated. Exactly-once execution cannot be guaranteed for a non-idempotent
  external action; those actions are not auto-replayed after an uncertain crash.
- Plans are bounded to six steps by default. Long projects should be split into
  smaller goals.
- Waiting jobs are not assumed approved. After changing permissions, credentials
  or policy, resume the job or create a new goal as appropriate.
- The `jobs` action is exposed through the existing voice/action tool interface;
  this change does not add a separate dashboard screen or alter the current HUD.
