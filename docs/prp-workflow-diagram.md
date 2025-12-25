# PRP Draft Workflow (Mermaid)

```mermaid
flowchart LR
    start([Start]) --> init[Initialize\ndetect root, load .env,\ncontext, agents]
    init --> submit[Submit Batch\nBatch API]
    submit --> poll{Poll Batch\nstatus?}
    poll -->|in_progress| poll
    poll -->|ended| process[Process Results\nparse + validate]
    poll -->|failed| fail([Fail])
    fail --> endNode([End])

    process --> check{Schema valid?}
    check -->|yes| save[Save draft\n+ raw]
    check -->|no| drop[Log & skip]

    save --> followup[Prepare Followup\nreal agents only]
    drop --> followup

    followup --> decision{Agents left?\npasses < max?}
    decision -->|yes| submit
    decision -->|no| compile[Compile drafts]
    compile --> success([Success])
    success --> endNode
```

Notes:
- Draft responses are validated against the strict Pydantic schema; invalid payloads are logged and excluded.
- Follow-up batches only include known agents (task-like IDs are dropped).
- Batch polling uses exponential backoff until `ended` or `failed`.
