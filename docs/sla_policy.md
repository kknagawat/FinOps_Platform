# Service Level Agreement (SLA) Policy

## Uptime Commitments

| Plan | Uptime SLA | Credit Threshold | Monthly Credit Cap |
|------|-----------|-------------------|-------------------|
| Free | No SLA | N/A | N/A |
| Basic | 99.0% | Below 99.0% | 10% of MRR |
| Pro | 99.5% | Below 99.5% | 25% of MRR |
| Growth | 99.9% | Below 99.9% | 30% of MRR |
| Scale | 99.9% | Below 99.9% | 50% of MRR |
| Enterprise | 99.95% | Below 99.95% | 100% of MRR |

## Support Response Times

### First Response SLA
| Priority | Basic | Pro | Growth/Scale | Enterprise |
|----------|-------|-----|-------------|------------|
| Critical | 24 hours | 4 hours | 1 hour | 15 minutes |
| High | 48 hours | 8 hours | 4 hours | 1 hour |
| Medium | 72 hours | 24 hours | 12 hours | 4 hours |
| Low | 5 business days | 48 hours | 24 hours | 8 hours |

### Resolution SLA
| Priority | Target Resolution Time | Maximum Resolution Time |
|----------|----------------------|------------------------|
| Critical | 4 hours | 8 hours |
| High | 12 hours | 24 hours |
| Medium | 48 hours | 72 hours |
| Low | 5 business days | 10 business days |

## SLA Credit Calculation
- Credits are calculated as a percentage of the monthly subscription fee.
- Credit percentage = (SLA target - actual uptime) / (100% - SLA target) * max credit percentage
- Credits are automatically applied to the next billing cycle.
- Credits do not roll over and cannot be converted to cash.
- Scheduled maintenance windows (announced 48 hours in advance) are excluded from uptime calculations.

## Exclusions
- Force majeure events
- Customer-caused outages (misconfiguration, exceeded rate limits)
- Third-party service failures outside our control
- Scheduled maintenance with 48-hour advance notice
- Beta or preview features

## Incident Communication
- Critical incidents: Status page update within 15 minutes, email notification within 30 minutes.
- Post-incident review (PIR) published within 5 business days for Critical/High incidents.
- Customers can subscribe to real-time status updates at status.finops-platform.com.

## Retention Offers for At-Risk Customers
- Customers who have experienced 2+ SLA breaches in a quarter qualify for a retention offer.
- Standard retention offer: 1 month free on current plan.
- Premium retention offer (Enterprise only): 2 months free + dedicated support engineer for 90 days.
- Retention offers must be approved by Customer Success Manager and documented in the CRM.
