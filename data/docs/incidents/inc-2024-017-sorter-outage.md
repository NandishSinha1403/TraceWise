# INC-2024-017: Sorter outage

- Severity: SEV1
- Date: 28 November 2024
- Duration: 3 hours 12 minutes (05:41 to 08:53)
- Incident commander: Devesh Rao

## Summary

Sorter could not connect to its PostgreSQL database, so no routes were produced for the morning. Drivers at all four depots could not start their routes in Tally. About 4,100 parcels were delivered one day late.

## Timeline

- 02:00 — Nightly batch plan started and failed to write routes. No alert fired, because the batch job reported success when it could not connect.
- 05:41 — Larrow Central depot staff reported empty routes in Tally. Primary on-call paged.
- 05:49 — Primary acknowledged (8 minutes).
- 06:20 — Cause found: the TLS certificate for Sorter's database connection had expired at 01:00.
- 07:35 — New certificate issued and deployed under the peak-freeze emergency exception.
- 08:53 — Routes generated and loaded into Tally. Drivers started late.

## Root cause

The certificate had a 1-year lifetime and was renewed by hand the previous year. No one was alerted before it expired. The batch job also treated a connection failure as an empty plan instead of an error.

## Follow-up actions

1. Monitor all internal certificates, alerting 30 days before expiry. Owner: Platform team. Done December 2024.
2. Make the batch job fail loudly and page on-call if it cannot connect. Owner: Routing team. Done December 2024.
3. Automate certificate renewal. Owner: Platform team. Done March 2025.

## Customer impact

Next Day and Same Day customers affected got an automatic refund of the delivery price through Ledger, without needing to claim.
