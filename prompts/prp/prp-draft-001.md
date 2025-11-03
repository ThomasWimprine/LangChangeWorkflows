---
allowed-tools: Task, Read, Grep, Glob, WebSearch, FileRead
argument-hint: [feature description]
description: Create an initial task list for agent concensus review
---

**IMPORTANT**: This is a DRAFT PRP process that calls multiple agents at once you are SPECIFICALLY being called for YOUR valued perspective input. If this should be delegated to another agent, please specify which one in the <delegation_suggestions> field of your response. That agent will be called next cycle if they haven't already been called. We are calling lots of agents if all defer to the 'security-reviewer' or 'devops-agent' we don't get 20 perspectives as intended but one perspective 20 times - which is wastefull and does not lead to the desired results. DO NOT DEFER AT THIS STAGE.

Create a rough draft PRP for initial review: $ARGUMENTS

**Purpose**: Quick requirements capture for "$ARGUMENTS"

**Process**: This is a lightweight first pass focusing on functionality and context without running extensive validation checks.

**Output**: Draft PRP files saved to `./prp/drafts/` ready for review and subsequent external steps to be brought to completion.

## Process

This command quickly drafts a PRP focusing on:
- Core functionality description
- Basic requirements gathering
- Initial context and scope
- High-level security/privacy considerations
- Accessibility and inclusion needs
- Life-critical development considerations (when applicable)
- Task decomposition and breakdown for complex features
- Preventing scope creep early

### Implementation Steps

1. **Create Draft**: Generate draft PRP following the template below
4. **Generate Report**:
	 - Next steps for user


## Life-Critical Standards Consideration

When drafting PRPs, consider if the feature involves:
- **Critical Systems**: Life-safety, security, or financial impact
- **Sensitive Data**: PII, PHI, financial, or confidential information
- **High Availability**: Systems requiring 99.9%+ uptime
- **Regulatory Compliance**: GDPR, HIPAA, SOX, PCI DSS requirements
- **Accessibility and Inclusion**: Features impacting diverse user needs

If YES to any above, the draft must include:
- **NIST Framework Applicability**: Which standards may apply (FIPS Excluded)
- **Emergency Stop Triggers**: What could halt development
- **Quality Gate Requirements**: Which validation checkpoints needed
- **Compliance Implications**: Regulatory requirements to consider 

**Note**: This is NOT the final PRP. It is a draft task list for review and refinement before running subsequent steps for full validation and completion.

## Task Decomposition and Agent Coordination

Leverage @Project Manager, @Architect and other agents as needed to **break down and define tasks** - NOT to solve them (solving happens in subsequent steps).

**Decomposition Process**:
- Break down complex features into smaller objectives and finally individual single tasks
- Each task should be atomic and independently testable and deliverable
- Include all clarifying questions in the draft for user responses
- **Formula**: Complex Feature → N Objectives → N Tasks → N PRPs → N Commits → Complete Feature (merged to main, all tests passing)
- **Preventing Scope Creep**: Strategies to maintain focus and avoid feature bloat
- **Testability**: Ensure each task has clear and testable acceptance criteria
- **User Story Mapping**: Align tasks with user stories for clarity

**Size and Scope Guidelines**:
- Each task PRP must be focused and concise, target single responsibility
- All template sections must be completed (not optional - see template below)
- Be concise but complete in each section using bullet points over paragraphs


## Draft PRP Template

```markdown
# DRAFT PRP: [Feature Name]
**Status**: DRAFT - Requires Review
**Created**: [Date]

## Overview
[Brief description of the feature/task based on user input]

## Functional Requirements
- [Core functionality needed]
- [User-facing features]
- [Integration points]

## Technical Context
- [Affected systems/services]
- [Technology stack involved]
- [Dependencies identified]

## Initial Security Considerations
- [Data sensitivity level]
- [Authentication needs]
- [Authorization requirements]

## Initial Privacy Considerations
- [PII involved?]
- [Data collection scope]
- [User consent needs]

## Scope Definition
- **In Scope**: [What's included]
- **Out of Scope**: [What's excluded]
- **Assumptions**: [Key assumptions]

## Success Criteria
- [Measurable outcomes]
- [Performance targets]
- [Quality metrics]

## Questions for Review
- [Clarifications needed]
- [Design decisions required]
- [Risk assessments needed]

## User Stories
### User Experience
- As a [user type], I want [goal] so that [benefit]
- User workflow: [step-by-step user journey]
- Acceptance criteria: [specific requirements for user satisfaction]

### Admin Experience
- As an admin, I need [capability] to [accomplish administrative task]
- Admin workflows: [administrative processes and procedures]
- Management requirements: [reporting, monitoring, configuration needs]

### Auditor Experience
- As an auditor, I must verify [compliance requirement] to ensure [regulatory standard]
- Audit trail requirements: [what needs to be logged, tracked, and reportable]
- Compliance checkpoints: [validation points and documentation needs]

### Customer Experience
- As a customer, I expect [service level] when [using the system]
- Service impact: [how changes affect customer experience and satisfaction]
- Support requirements: [help desk, documentation, training needs]

### 3rd Party Experience (Family, Friend, etc)
- As a [family member/friend/partner], I need [access/information] to [support user]
- External integration: [third-party system requirements and API needs]
- Communication needs: [notifications, updates, status information]

## Stakeholder Impact Analysis
### Internal Stakeholders
- **Development Team**: Implementation complexity, technical debt, maintenance burden
- **Operations Team**: Deployment requirements, monitoring needs, support procedures
- **Security Team**: Threat model implications, access controls, audit requirements
- **Compliance Team**: Regulatory adherence, documentation needs, reporting requirements
- **Product Team**: Feature alignment, user research validation, success metrics

### External Stakeholders
- **Customer Communication**: Change notifications, training materials, support updates
- **Partner Integration**: API changes, integration testing, coordination requirements
- **Regulatory Bodies**: Compliance notifications, audit preparation, documentation
- **Public Disclosure**: Security announcements, privacy policy updates, transparency reports

## User Journey Considerations
### Onboarding Experience
- Initial setup and configuration requirements
- Training and education needs
- First-time user success criteria

### Day-to-Day Usage Experience
- Routine workflows and common tasks
- Performance and reliability expectations
- Integration with existing user habits

### Error/Exception Experience
- Error handling and recovery procedures
- Support escalation paths
- User communication during issues

### Offboarding Experience
- Data export and migration requirements
- Access revocation procedures
- Knowledge transfer needs

### Emergency/Crisis Experience
- System failure scenarios and backup procedures
- Emergency contact and communication protocols
- Business continuity requirements

## Accessibility and Inclusion Requirements
### Visual Accessibility
- Screen reader compatibility and ARIA labels
- Color contrast and visual design considerations
- Font size and readability requirements

### Motor Accessibility
- Keyboard navigation and shortcuts
- Touch target sizes and gesture alternatives
- Alternative input method support

### Cognitive Accessibility
- Clear language and simplified workflows
- Consistent navigation and predictable behavior
- Error prevention and clear feedback

### Language and Cultural Considerations
- Internationalization and localization needs
- Cultural sensitivity in design and content
- Multi-language support requirements
```

## Workflow

Step 001 — Panel Decomposition (TASK001)

What you should do:
- Read the feature description: "$ARGUMENTS".
- Produce an initial task decomposition using the TARGET JSON TEMPLATE for TASK001.
- Keep tasks atomic: single objective, ≤2 affected components, ≤1 dependency (prefer 0).
- Provide clear, testable acceptance criteria for each task (3–5 bullets).
- Include clarifying questions where needed; do not implement solutions.
- Return JSON only (no prose outside JSON). Follow the exact key names and structure.

Agent usage and delegation suggestions:
- Only suggest or delegate to agents that exist in the provided Agent Catalog (JSON) included in system context.
- Use the exact agent id values from available_agents[].id when adding delegation_suggestions.
- Reference repository: local Claude Agents registry at ~/.claude/agents (plus any paths in CLAUDE_AGENT_DIRS). Each agent’s .md defines its system role.

Determinism and structure:
- Do not add or remove keys from the template; preserve order and nesting.
- Do not invent agent names. If an appropriate agent is not listed, omit it.
- Keep language concise and consistent to support stable caching across runs.


## Example Usage

```
/draft-prp "Implement two-factor authentication for admin users"
```

Creates a draft covering:
- Basic 2FA flow
- Admin user context
- Initial security needs
- Questions about SMS vs TOTP
- Scope boundaries

## Benefits

- **Fast**: Get initial requirements documented quickly
- **Iterative**: Refine before full analysis
- **Collaborative**: Review points built in
- **Efficient**: Avoid rework by clarifying early

Remember: This is step 1 of many in your PRP process. Always review specifications, documentation, validate and refine before proceeding.


### Best Practices
- **Single Responsibility**: Each PRP should address one focused feature or fix
- **Clear Description**: Use descriptive hyphenated names that explain the feature
- **Sequential Numbering**: Use next available number in sequence
- **Letter Variants**: Use 'a' for first attempt, 'b' for revision if needed

**Note**: Keep PRPs focused and single-purpose. Complex features can be broken into multiple sequential PRPs rather than trying to handle everything in one. The system will automatically handle splitting during generation.