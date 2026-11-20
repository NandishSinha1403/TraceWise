# Internal Systems Overview

Brackenfield's engineering team (about 35 people, led by CTO Devesh Rao) builds and runs four main systems.

## Tally — driver app

- Mobile app for Android, used by all drivers and cargo bike riders.
- Shows routes from Sorter, records scans and proof of delivery, runs the 12-item pre-shift check.
- Works offline; scans sync when the phone reconnects. Unsynced scans are kept for 72 hours on the phone.
- Introduced in 2018.

## Sorter — routing and sorting

- Plans van routes and assigns parcels to sorting bays.
- Live since August 2021. Written in Python, stores routes in PostgreSQL.
- Runs a nightly batch plan at 02:00 and intraday re-plans every 30 minutes from 07:00 to 19:00 (see ADR-007).
- Owner: Routing team.

## Ledger — billing

- Invoices business accounts, applies discounts and surcharges, pays out approved claims as account credits, and handles staff expense claims.
- Live since January 2022. Uses PostgreSQL (see ADR-003).
- Invoices are issued on the 1st of each month for the previous month.
- Owner: Payments team.

## Beacon — tracking API

- Public REST API that business customers use to track parcels and receive status webhooks.
- v2 launched in March 2024; v1 was switched off on 31 January 2025.
- Owner: Platform team.

## Shared infrastructure

- All services emit traces through OpenTelemetry.
- Certificates for internal database connections are issued by the internal certificate authority and are valid for 1 year. Since INC-2024-017 they are monitored, with alerts 30 days before expiry.
- Deploys happen Monday to Thursday only. There are no deploys on Fridays, weekends, or during the peak freeze from 20 November to 31 December.
