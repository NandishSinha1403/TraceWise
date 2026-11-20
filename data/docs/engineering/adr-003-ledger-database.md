# ADR-003: Use PostgreSQL for Ledger

- Status: Accepted
- Date: 2021-06-14
- Deciders: Devesh Rao, Payments team

## Context

Ledger, the new billing system, needs to store invoices, account balances, discounts, and claim credits. Money movements must never be partly applied. The team considered PostgreSQL, a managed document database, and continuing to use spreadsheets exported from the old accounting tool.

## Decision

Use PostgreSQL 13 as Ledger's only database.

## Reasons

- Transactions: crediting a claim and updating an invoice must succeed or fail together. PostgreSQL gives this with standard transactions.
- The team already ran PostgreSQL for Sorter, so no new operational skills were needed.
- Amounts are stored as integer pence, never as floating-point numbers, to avoid rounding errors.

## Rejected options

- Document database: rejected because multi-document transactions were harder to get right and the team had no experience running it.
- Spreadsheets: rejected because they could not support more than about 300 accounts.

## Consequences

- Nightly backups are kept for 35 days, and point-in-time recovery is enabled.
- Ledger was upgraded to PostgreSQL 15 in 2024.
