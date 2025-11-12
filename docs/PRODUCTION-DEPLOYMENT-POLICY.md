# Production Deployment Policy

**Effective Date**: 2025-11-07
**Environment Status**: 🔴 **PRODUCTION** 🔴
**Policy Version**: 1.0

---

## Overview

This Kubernetes cluster has graduated from development/lab to **PRODUCTION STATUS**. All deployments, configuration changes, and infrastructure modifications MUST follow this policy to maintain system stability, security, and compliance.

---

## Core Principles

### 1. Zero Direct Access
**NO direct deployments, modifications, or `kubectl apply` commands allowed in production.**

All changes MUST:
- Go through pull request workflow
- Pass CI/CD validation gates
- Receive code review approval
- Be deployed via automated pipeline

### 2. Read-Only (RO) Access
**Production cluster access is READ-ONLY for all users.**

Allowed operations:
- ✅ `kubectl get` (read resources)
- ✅ `kubectl describe` (inspect resources)
- ✅ `kubectl logs` (view logs)
- ✅ `kubectl exec -it` (debugging, read-only)

Blocked operations:
- ❌ `kubectl apply/create/delete` (use CI/CD)
- ❌ `kubectl edit` (use PR workflow)
- ❌ `kubectl patch` (use PR workflow)
- ❌ `talosctl apply-config` (use CI/CD)
- ❌ Direct Terraform apply (use CI/CD)

### 3. Test-Driven Development (TDD)
**ALL infrastructure and application code MUST follow TDD methodology.**

Required pattern:
1. **RED**: Write failing tests FIRST
2. **GREEN**: Implement minimal code to pass tests
3. **REFACTOR**: Improve code quality

Git history MUST show RED → GREEN → REFACTOR commits.

---

## Deployment Workflow

### Standard Deployment Process

```
┌─────────────────────────────────────┐
│ 1. Create Feature Branch            │
│    git checkout -b feature/xxx       │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 2. Write Tests FIRST (TDD RED)      │
│    - Write failing tests             │
│    - Commit: test: Add tests (RED)  │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 3. Implement Code (TDD GREEN)       │
│    - Minimal implementation          │
│    - Commit: feat: Implement (GREEN)│
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 4. Refactor (TDD REFACTOR)          │
│    - Improve code quality            │
│    - Commit: refactor: Improve      │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 5. Push to Remote                    │
│    git push origin feature/xxx       │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 6. Create Pull Request              │
│    - Comprehensive description       │
│    - Link to PRP/issue               │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 7. CI/CD Validation (Automated)     │
│    ✅ TDD pattern verification      │
│    ✅ Test coverage (100% required) │
│    ✅ Security scan (tfsec/trivy)   │
│    ✅ Terraform validate             │
│    ✅ Integration tests              │
│    ✅ Performance benchmarks         │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 8. Code Review                       │
│    - Minimum 1 approval required     │
│    - Security review for infra       │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 9. Merge to Main                     │
│    - Squash or merge commit          │
│    - Auto-delete feature branch      │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│ 10. Automated Deployment             │
│     - CI/CD pipeline deploys         │
│     - Health checks validate         │
│     - Automatic rollback on failure  │
└─────────────────────────────────────┘
```

---

## CI/CD Quality Gates

### Gate 1: TDD Cycle Verification
**Requirement**: RED-GREEN-REFACTOR pattern in git history

**Validation**:
```bash
# Check commit messages show TDD pattern
git log --oneline --grep="test:" --grep="feat:" --grep="refactor:"

# Verify tests committed before implementation
scripts/tdd-enforcement/verify-tdd-cycle.sh
```

**Failure Action**: PR blocked, developer must fix commit history

---

### Gate 2: Test Coverage
**Requirement**: 100% coverage (lines, branches, functions)

**Validation**:
```bash
# Application code
pytest --cov=src --cov-report=term-missing --cov-fail-under=100

# Infrastructure code (Terratest)
go test -v ./tests/terraform/ -coverprofile=coverage.out
go tool cover -func=coverage.out
```

**Failure Action**: PR blocked until coverage reaches 100%

---

### Gate 3: Security Scanning

**Requirements**:
- Zero CRITICAL vulnerabilities
- Zero HIGH vulnerabilities
- MEDIUM vulnerabilities must have mitigation plan

**Tools**:
```bash
# Terraform security
tfsec terraform/ --minimum-severity MEDIUM
checkov -d terraform/

# Container security
trivy image <image>:<tag> --severity CRITICAL,HIGH

# Dependency scanning
pip-audit --strict (Python)
npm audit --audit-level=high (Node.js)
```

**Failure Action**: PR blocked until vulnerabilities resolved

---

### Gate 4: Infrastructure Validation

**Terraform**:
```bash
terraform init
terraform validate
terraform fmt -check -recursive
terraform plan -out=tfplan
```

**Kubernetes Manifests**:
```bash
kubectl apply --dry-run=client -f kubernetes/
kubeval kubernetes/**/*.yaml
```

**Failure Action**: PR blocked until validation passes

---

### Gate 5: Integration Tests

**Requirements**:
- All critical paths tested end-to-end
- Network connectivity validated
- Service discovery tested
- Performance benchmarks met

**Example**:
```bash
# VPN tunnel connectivity
./tests/integration/test_vpn_connectivity.sh

# Kubernetes cluster health
./tests/integration/test_cluster_health.sh

# Application endpoints
./tests/integration/test_application_endpoints.sh
```

**Failure Action**: PR blocked until integration tests pass

---

### Gate 6: Performance Benchmarks

**Requirements**:
- Latency SLOs met (p95 <50ms for VPN, p99 <100ms)
- Throughput meets targets (>100 Mbps for VPN)
- Resource utilization within limits

**Validation**:
```bash
# Latency test
./scripts/monitor-vpn-latency.sh --validate

# Load test
hey -z 60s -c 10 https://api.example.com/health
```

**Failure Action**: PR blocked or warning (depends on severity)

---

## Access Control Matrix

| Role | Read Access | Write Access | Emergency Access |
|------|-------------|--------------|------------------|
| **Developer** | ✅ kubectl get/describe/logs | ❌ NONE (use PR) | ❌ NONE |
| **DevOps** | ✅ kubectl get/describe/logs | ❌ NONE (use PR) | ⚠️ Break-glass only |
| **Security** | ✅ Full read access | ❌ NONE (use PR) | ⚠️ Break-glass only |
| **CI/CD Pipeline** | ✅ Full read access | ✅ Automated deploy | N/A (automated) |

---

## Emergency Procedures

### Break-Glass Access

**When to Use**:
- Production incident requiring immediate intervention
- Security breach requiring urgent response
- Complete CI/CD pipeline failure

**Procedure**:
1. **Incident Declaration**: File incident ticket with severity
2. **Approval Required**: Obtain approval from 2 of 3 maintainers
3. **Access Grant**: Temporary elevated access (max 4 hours)
4. **Audit Log**: All actions logged and reviewed
5. **Post-Incident Review**: Mandatory review within 24 hours
6. **Retroactive PRP**: Create PRP documenting changes made

**Example**:
```bash
# Request break-glass access
./scripts/request-emergency-access.sh --incident=INC-12345 --duration=4h

# Access granted temporarily
# Make necessary changes
kubectl apply -f emergency-fix.yaml

# Access automatically revoked after 4 hours
# Post-incident PRP required within 24 hours
```

---

## Rollback Procedures

### Automated Rollback

**Triggers**:
- Health check failures after deployment
- Increased error rate (>1% 5xx responses)
- Performance degradation (p95 latency >2x baseline)
- Resource exhaustion (CPU/memory >90%)

**Process**:
```bash
# CI/CD automatically detects failure
# Rolls back to previous version
# Notifies team via Slack/PagerDuty

# Manual rollback (if needed)
kubectl rollout undo deployment/<name>
terraform apply -var-file=previous.tfvars
```

### Manual Rollback

**When to Use**:
- Automated rollback failed
- Partial rollback needed (specific component)
- Coordinated rollback across multiple services

**Procedure**:
1. Create rollback PR with revert commits
2. Fast-track through CI/CD (skip some gates if approved)
3. Deploy via automated pipeline
4. Validate rollback successful
5. Root cause analysis within 48 hours

---

## Compliance and Auditing

### Audit Requirements

**All deployments must maintain audit trail**:
- Who requested the change (PR author)
- What was changed (git diff, Terraform plan)
- When it was deployed (CI/CD timestamp)
- Why it was changed (PRP reference, issue link)
- How it was validated (test results, security scan)

**Audit Log Retention**: 1 year minimum

**Example Audit Entry**:
```yaml
deployment_id: "deploy-2025-11-07-006a"
timestamp: "2025-11-07T14:23:45Z"
pr_number: 42
author: "thomas"
reviewer: "security-team"
prp_reference: "WU-006a-vpn-tunnel-foundation"
changes:
  - resource: "google_compute_instance.vpn_gateway"
    action: "create"
    validation: "tfsec PASS, terratest PASS"
  - resource: "google_compute_firewall.allow_ipsec"
    action: "create"
    validation: "security review APPROVED"
deployment_result: "SUCCESS"
rollback_available: true
```

---

## Monitoring and Alerting

### Required Monitoring

**All deployments must include**:
1. **Health Checks**: Liveness and readiness probes
2. **Metrics Export**: Prometheus metrics endpoint
3. **Logging**: Structured logs to centralized system
4. **Alerting**: Critical alerts to PagerDuty/Slack

**Example**:
```yaml
# Deployment includes monitoring
apiVersion: v1
kind: Service
metadata:
  name: vpn-gateway-metrics
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"

# Alert rules defined
groups:
  - name: vpn-tunnel
    rules:
      - alert: VPNTunnelDown
        expr: up{job="vpn-gateway"} == 0
        for: 5m
        annotations:
          summary: "VPN tunnel is down"
          runbook: "docs/runbooks/vpn-tunnel-down.md"
```

---

## Testing Requirements

### Test Pyramid for Production

```
┌─────────────────────────────────────┐
│   E2E Tests (10%)                    │  ← Critical user journeys
│   - Full deployment validation       │
│   - Cross-service integration        │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│   Integration Tests (30%)            │  ← Service boundaries
│   - API contracts                    │
│   - Network connectivity             │
│   - Database interactions            │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│   Unit Tests (60%)                   │  ← Component logic
│   - Functions and methods            │
│   - Terraform resources              │
│   - Business logic                   │
└─────────────────────────────────────┘
```

**Coverage Requirements**:
- Unit tests: 100% coverage
- Integration tests: All critical paths
- E2E tests: Top 5 user journeys
- Performance tests: SLO validation
- Security tests: OWASP Top 10

---

## Change Management

### Change Categories

| Category | Approval | Testing | Deployment Window |
|----------|----------|---------|-------------------|
| **CRITICAL** (Security fix) | 2 approvers | Full suite + manual | Immediate (emergency) |
| **HIGH** (Infrastructure) | 1 approver | Full suite | Scheduled maintenance |
| **MEDIUM** (Feature) | 1 approver | Full suite | Any time (automated) |
| **LOW** (Documentation) | Self-approve | Docs build | Any time |

### Maintenance Windows

**Scheduled Maintenance**:
- **Weekly**: Tuesday 02:00-04:00 UTC (low traffic)
- **Monthly**: First Sunday 00:00-06:00 UTC (major changes)

**Emergency Maintenance**: As needed with incident approval

---

## Documentation Requirements

### Required Documentation

Every deployment MUST include:

1. **PRP (Product Requirements Proposal)**:
   - Business justification
   - Technical architecture
   - Security assessment
   - Test strategy

2. **Runbooks**:
   - Deployment procedure
   - Rollback procedure
   - Troubleshooting guide
   - Monitoring dashboard

3. **Architecture Diagrams**:
   - System design
   - Network topology
   - Data flow
   - Integration points

4. **Test Documentation**:
   - Test coverage report
   - Integration test results
   - Performance benchmark results
   - Security scan results

---

## Exceptions and Waivers

### Requesting an Exception

**Process**:
1. Create exception request issue with justification
2. Document risk assessment and mitigation plan
3. Obtain approval from 2 of 3 maintainers
4. Set expiration date (max 30 days)
5. Create follow-up PRP to remove exception

**Valid Reasons**:
- Vendor limitation (third-party API)
- Temporary workaround for critical bug
- Pilot/experiment in isolated namespace

**Invalid Reasons**:
- Time pressure ("need to ship fast")
- Lack of test infrastructure
- "It's just a small change"
- "Trust me, it works"

---

## Summary

**Key Takeaways**:
1. ✅ **All deployments via CI/CD** - No direct access
2. ✅ **Read-only production access** - Use PR workflow for changes
3. ✅ **TDD mandatory** - Tests before code, always
4. ✅ **100% test coverage** - No exceptions
5. ✅ **Security scanning** - Zero critical/high vulnerabilities
6. ✅ **Full audit trail** - Every change tracked

**Non-Compliance Consequences**:
- PR blocked by automated gates
- Access revoked for repeated violations
- Incident review for production issues

---

**Document Owner**: Platform Team
**Review Schedule**: Quarterly
**Next Review**: 2026-02-07

*This policy applies to ALL production infrastructure and applications. Violations will result in automated enforcement and incident review.*
