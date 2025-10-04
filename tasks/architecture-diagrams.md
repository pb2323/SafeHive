# SafeHive AI Security Sandbox - Architecture Diagrams & Workflows

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SafeHive AI Security Sandbox                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   CLI Interface │    │  Human-in-the-  │    │   Metrics &     │             │
│  │   (typer+rich)  │◄──►│   Loop Controls │◄──►│   Monitoring    │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        Sandbox Orchestrator                                │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │ │
│  │  │  Orchestrator   │  │    User Twin    │  │   Agent Memory  │             │ │
│  │  │   Agent         │  │    Agent        │  │   Management    │             │ │
│  │  │  (LangChain)    │  │  (LangChain)    │  │   (LangChain)   │             │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        Security Guards Layer                               │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │ │
│  │  │   Privacy   │ │    Task     │ │   Prompt    │ │   MCP       │           │ │
│  │  │   Sentry    │ │  Navigator  │ │ Sanitizer   │ │   Server    │           │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        Vendor Agents Layer                                 │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │ │
│  │  │   Honest        │  │   Malicious     │  │   Vendor        │             │ │
│  │  │   Vendor        │  │   Vendor        │  │   Factory       │             │ │
│  │  │  (LangChain)    │  │  (LangChain)    │  │   (Personality  │             │ │
│  │  │                 │  │                 │  │    Management)  │             │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        External Systems                                    │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │ │
│  │  │   Ollama        │  │   Decoy Data    │  │   Incident      │             │ │
│  │  │   (AI Models)   │  │   Generator     │  │   Logging       │             │ │
│  │  │                 │  │   (Faker)       │  │   (Structured)  │             │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Agent Communication Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───►│  Orchestrator   │───►│   User Twin     │
│  (Food Order)   │    │    Agent        │    │    Agent        │
└─────────────────┘    │  (LangChain)    │    │  (LangChain)    │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Security       │    │  Constraint     │
                       │  Guards         │    │  Validation     │
                       │  (4 Guards)     │    │  (Task Nav.)    │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Vendor        │
                       │   Selection     │
                       │  (Honest/Mal.)  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Vendor        │
                       │   Agent         │
                       │  (LangChain)    │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Response      │
                       │   Processing    │
                       │  (Guards Check) │
                       └─────────────────┘
```

## Attack Detection & MCP Integration Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Malicious      │───►│  Attack         │───►│  Threshold      │
│  Vendor         │    │  Detection      │    │  Manager        │
│  (LangChain)    │    │  (Regex/ML)     │    │  (Per-IP)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Attack         │    │  Threshold      │
                       │  Pattern        │    │  Exceeded?      │
                       │  (SQLi/XSS/PT)  │    │  (3 attempts)   │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Normal         │    │  MCP Server     │
                       │  Response       │    │  Activation     │
                       │  (Block/Allow)  │    │  (Decoy Data)   │
                       └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Stakeholder    │
                                               │  Alert          │
                                               │  (Incident Log) │
                                               └─────────────────┘
```

## Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User          │───►│   CLI           │───►│  Orchestrator   │
│   Interface     │    │   Interface     │    │   Agent         │
│   (Commands)    │    │   (typer+rich)  │    │  (LangChain)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Request        │
                                               │  Processing     │
                                               │  (Standardized) │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Security       │
                                               │  Guards         │
                                               │  (4 Guards)     │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Vendor         │
                                               │  Communication │
                                               │  (LangChain)    │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Response       │
                                               │  Processing     │
                                               │  (Guards Check) │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Human-in-the-  │
                                               │  Loop Controls  │
                                               │  (Approve/etc.) │
                                               └─────────────────┘
```

## Component Interaction Matrix

| Component | Orchestrator | User Twin | Guards | Vendors | CLI | Memory |
|-----------|-------------|-----------|--------|---------|-----|--------|
| **Orchestrator** | - | Direct | Direct | Direct | Direct | Direct |
| **User Twin** | Direct | - | - | - | - | Direct |
| **Guards** | Direct | - | - | Intercept | - | - |
| **Vendors** | Direct | - | Intercept | - | - | Direct |
| **CLI** | Direct | - | - | - | - | - |
| **Memory** | Direct | Direct | - | Direct | - | - |

## Security Guard Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Security Guards Layer                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   Privacy       │    │    Task         │    │   Prompt        │             │
│  │   Sentry        │    │  Navigator      │    │  Sanitizer      │             │
│  │                 │    │                 │    │                 │             │
│  │ • PII Detection │    │ • Constraint    │    │ • Input         │             │
│  │ • Data Masking  │    │   Enforcement   │    │   Validation    │             │
│  │ • Over-sharing  │    │ • Budget Limits │    │ • Malicious     │             │
│  │   Prevention    │    │ • Dietary       │    │   Pattern       │             │
│  │                 │    │   Restrictions  │    │   Detection     │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        MCP Server Integration                              │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │ │
│  │  │  Attack         │  │  Threshold      │  │  Decoy Data     │             │ │
│  │  │  Detection      │  │  Management     │  │  Generation     │             │ │
│  │  │                 │  │                 │  │                 │             │ │
│  │  │ • SQL Injection │  │ • Per-IP        │  │ • Fake Credit   │             │ │
│  │  │ • XSS Attacks   │  │   Tracking      │  │   Cards         │             │ │
│  │  │ • Path          │  │ • Attempt       │  │ • Fake Orders   │             │ │
│  │  │   Traversal     │  │   Counting      │  │ • Fake Profiles │             │ │
│  │  │ • Pattern       │  │ • Threshold     │  │ • Realistic     │             │ │
│  │  │   Matching      │  │   Enforcement   │  │   Data          │             │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Agent Memory & State Management

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Agent Memory Architecture                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │  Orchestrator   │    │    User Twin    │    │   Vendor        │             │
│  │   Memory        │    │    Memory       │    │   Memory        │             │
│  │                 │    │                 │    │                 │             │
│  │ • Conversation  │    │ • Preferences   │    │ • Attack        │             │
│  │   History       │    │ • Constraints   │    │   Patterns      │             │
│  │ • Order Context │    │ • Decision      │    │ • Customer      │             │
│  │ • Vendor        │    │   History       │    │   Interactions  │             │
│  │   Interactions  │    │ • Learning      │    │ • Personality   │             │
│  │ • Learning      │    │   Patterns      │    │   State         │             │
│  │   Patterns      │    │                 │    │                 │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        LangChain Memory System                             │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │ │
│  │  │  Conversation   │  │  Summary        │  │  Buffer         │             │ │
│  │  │  Buffer         │  │  Memory         │  │  Memory         │             │ │
│  │  │                 │  │                 │  │                 │             │ │
│  │  │ • Full History  │  │ • Condensed     │  │ • Recent        │             │ │
│  │  │ • Context       │  │   Context       │  │   Interactions  │             │ │
│  │  │ • Timestamps    │  │ • Key Points    │  │ • Active        │             │ │
│  │  │ • Metadata      │  │ • Patterns      │  │   Context       │             │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## MCP Server Integration Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Orchestrator  │───►│   MCP Server    │───►│   DoorDash      │
│   Agent         │    │   Interface     │    │   API           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Security      │    │   Credential    │    │   Live Order    │
│   Guards        │    │   Management    │    │   Processing    │
│                 │    │                 │    │                 │
│ • Privacy       │    │ • API Keys      │    │ • Order         │
│ • Task Nav      │    │ • Sandbox Mode  │    │   Validation    │
│ • Prompt San    │    │ • Live Mode     │    │ • Payment       │
└─────────────────┘    └─────────────────┘    │ • Delivery      │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Order         │
                                               │   Confirmation  │
                                               │   & Tracking    │
                                               └─────────────────┘
```

### MCP Server Flow
1. **Testing Mode**: Orchestrator communicates with vendor agents (simulated)
2. **Live Mode**: Orchestrator communicates with MCP server (real DoorDash API)
3. **Security Guards**: All communications pass through security guards regardless of mode
4. **Credential Management**: Secure handling of DoorDash API credentials
5. **Order Validation**: Safety checks before placing real orders
6. **Confirmation**: Real order tracking and delivery updates

## Deployment & Configuration Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Configuration │───►│   Agent         │───►│   Runtime       │
│   (YAML)        │    │   Initialization│    │   Execution     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Guard          │    │  Agent          │
                       │  Configuration  │    │  Communication  │
                       │  (Enable/Disable│    │  (LangChain)    │
                       │   Thresholds)   │    │                 │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Vendor         │    │  Security       │
                       │  Personality    │    │  Monitoring     │
                       │  Setup          │    │  (Real-time)    │
                       └─────────────────┘    └─────────────────┘
```

## Key Design Decisions

### 1. **Single Framework (LangChain Only)**
- **Rationale**: Simplified architecture, consistent patterns, easier maintenance
- **Benefits**: Single learning curve, unified testing, better integration
- **Trade-offs**: None significant for this use case

### 2. **Agent Memory Isolation**
- **Rationale**: Security requirement to prevent cross-agent data leakage
- **Implementation**: Separate LangChain memory instances per agent
- **Benefits**: Enhanced security, better debugging, isolated learning

### 3. **Guard-Based Security**
- **Rationale**: Intercept and analyze all agent communications
- **Implementation**: Four specialized guards with different responsibilities
- **Benefits**: Comprehensive security coverage, modular design

### 4. **MCP Server Integration**
- **Rationale**: Enable live DoorDash ordering for successful demonstrations
- **Implementation**: Secure API integration with sandbox and live modes
- **Benefits**: Real-world validation, compelling demos, separation of testing vs live ordering

### 5. **Human-in-the-Loop Controls**
- **Rationale**: Allow human oversight and decision-making
- **Implementation**: CLI-based approval/redaction/quarantine options
- **Benefits**: Enhanced security, educational value, stakeholder control

## Performance Characteristics

| Component | Response Time | Memory Usage | CPU Usage |
|-----------|---------------|--------------|-----------|
| **CLI Interface** | < 100ms | Low | Low |
| **Orchestrator Agent** | < 2s | Medium | Medium |
| **Security Guards** | < 1s | Low | Low |
| **Vendor Agents** | < 2s | Medium | Medium |
| **MCP Server** | < 5s | Low | Low |
| **Memory Operations** | < 500ms | Medium | Low |

## Scalability Considerations

- **Horizontal Scaling**: Each agent can run on separate processes/containers
- **Memory Management**: LangChain memory can be persisted to external storage
- **Guard Performance**: Guards are stateless and can be replicated
- **Model Loading**: Ollama models can be shared across agent instances
- **Configuration**: YAML-based configuration allows runtime updates

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Purpose**: Architecture documentation for SafeHive AI Security Sandbox  
**Audience**: Development team, stakeholders, presentation materials
