# On-call and Incident Severity

## Rotation

- Engineering on-call rotates weekly. Handover is on Monday at 09:00.
- Each week has a primary and a secondary on-call engineer.
- If the primary does not acknowledge a page in time, the secondary is paged automatically.
- On-call engineers get an allowance of £150 per week, plus time off in lieu for any incident work between 22:00 and 07:00.

## Severity levels

| Level | Meaning | Acknowledge within |
|---|---|---|
| SEV1 | Deliveries stopped across the network, data loss, or a security breach | 10 minutes |
| SEV2 | One depot or one system badly degraded | 30 minutes |
| SEV3 | Minor problem with a workaround | Next business day |

## SEV1 process

1. The primary acknowledges and opens an incident channel.
2. An incident commander is named. For SEV1 this must be an engineering manager or the CTO.
3. The COO is told within 30 minutes so depots can be warned.
4. Status updates go out every 30 minutes until resolved.

## Postmortems

- Every SEV1 and SEV2 needs a written, blameless postmortem within 5 business days.
- Each postmortem lists follow-up actions with an owner and a due date.
- Postmortem IDs have the form INC-<year>-<number>.
