# Escalation Procedures

## Overview
This document defines the escalation procedures for support tickets, billing disputes, and technical incidents.

## Support Ticket Escalation

### Automatic Escalation Triggers
1. **Time-based**: Any ticket exceeding its priority's resolution SLA is automatically escalated.
2. **Customer-requested**: Customer explicitly requests escalation (flag in ticket system).
3. **Sentiment-based**: Two or more negative ratings (1-2 stars) from the same customer within 30 days.
4. **Revenue-based**: Any ticket from a customer with ARR > $50,000 and priority >= High is auto-escalated to Tier 2.

### Escalation Tiers
| Tier | Handler | Scope | Response SLA |
|------|---------|-------|-------------|
| Tier 1 | Support Agent | Standard troubleshooting, FAQ, basic config | Per plan SLA |
| Tier 2 | Senior Support / Team Lead | Complex issues, multi-system problems | 50% of standard SLA |
| Tier 3 | Engineering Team | Bug fixes, infrastructure issues | 4 hours (Critical), 24 hours (High) |
| Tier 4 | VP Engineering + Product | Systemic issues, feature escalations | Immediate for Critical |

### Escalation Procedure
1. Agent documents all troubleshooting steps taken.
2. Agent selects escalation reason from predefined list.
3. System notifies the next-tier handler and the customer.
4. Escalated ticket retains full conversation history.
5. Receiving handler must acknowledge within 30 minutes during business hours.

## Billing Dispute Escalation
- Disputes under $500: Resolved by Billing Support (Tier 1).
- Disputes $500-$5,000: Escalated to Billing Manager.
- Disputes above $5,000: Escalated to Finance Director.
- All disputes must be acknowledged within 2 business days.
- Resolution target: 10 business days from acknowledgment.

## Technical Incident Escalation
- **SEV-1 (Critical)**: All hands. Engineering lead, VP Engineering, and on-call SRE notified immediately. War room opened. Customer communication every 30 minutes.
- **SEV-2 (High)**: Engineering lead and on-call SRE. Customer communication every 2 hours.
- **SEV-3 (Medium)**: Assigned to engineering team backlog. Customer updated daily.
- **SEV-4 (Low)**: Tracked in backlog. No active escalation required.

## De-escalation Criteria
A ticket can be de-escalated when:
1. Root cause is identified AND
2. A fix or workaround is deployed AND
3. Customer confirms the issue is resolved OR
4. 48 hours pass with no further customer response after fix deployment.

## Escalation Metrics (Tracked Monthly)
- Escalation rate by tier
- Average time-to-escalation
- Escalation resolution rate (% resolved at each tier)
- Customer satisfaction post-escalation (CSAT score)
- Repeat escalation rate (same customer, same issue category within 90 days)
