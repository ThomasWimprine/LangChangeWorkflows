# Multi-Language Project Support

The LangGraph PRP Workflow supports **any project type** - Python, Node.js, Go, Terraform, Rust, Java, etc.

## Quick Start for Any Project

```bash
# From any project directory (Node, Go, Terraform, etc.)
cd /path/to/your/project

# Auto-detect and run
python3 ~/Repositories/LangChainWorkflows/examples/auto_detect_runner.py
```

The auto-detect runner will:
1. ✅ Detect your project type (Node, Go, Python, Terraform, etc.)
2. ✅ Find your source and test directories automatically
3. ✅ Run the appropriate coverage tools
4. ✅ Apply correct validation rules

## Supported Project Types

| Type | Detection | Source Dirs | Test Dirs | Coverage Tool |
|------|-----------|-------------|-----------|---------------|
| **Node.js** | `package.json` | `src/`, `lib/` | `tests/`, `__tests__/` | `jest`, `vitest` |
| **Go** | `go.mod` | `cmd/`, `internal/`, `pkg/` | `*_test.go` | `go test -cover` |
| **Python** | `setup.py`, `pyproject.toml` | `src/`, `lib/`, `app/` | `tests/` | `pytest --cov` |
| **Terraform** | `*.tf` | `modules/`, `*.tf` | `tests/`, `examples/` | `terratest`, `tfsec` |
| **Rust** | `Cargo.toml` | `src/` | `tests/` | `cargo tarpaulin` |
| **Java** | `pom.xml`, `build.gradle` | `src/main/java` | `src/test/java` | `jacoco` |

## Project-Specific Examples

### Node.js Project

```bash
# Your project structure
my-node-app/
├── package.json
├── src/
│   └── index.ts
├── tests/
│   └── index.test.ts
└── jest.config.js

# Run workflow
cd my-node-app
python3 ~/Repositories/LangChainWorkflows/examples/auto_detect_runner.py
```

**What it validates:**
- ✅ 100% test coverage via Jest/Vitest
- ✅ No mocks in `src/` (only in `tests/`)
- ✅ All TypeScript/JavaScript tests passing

### Go Project

```bash
# Your project structure (standard Go layout)
my-go-app/
├── go.mod
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   └── handler/
│       ├── handler.go
│       └── handler_test.go
└── pkg/
    └── utils/
        ├── utils.go
        └── utils_test.go

# Run workflow
cd my-go-app
python3 ~/Repositories/LangChainWorkflows/examples/auto_detect_runner.py
```

**What it validates:**
- ✅ 100% test coverage via `go test -cover`
- ✅ No mock libraries in production code
- ✅ All `*_test.go` tests passing

### Terraform Project

```bash
# Your project structure
my-terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── modules/
│   └── vpc/
│       └── main.tf
└── tests/
    └── terraform_test.go

# Run workflow
cd my-terraform
python3 ~/Repositories/LangChainWorkflows/examples/auto_detect_runner.py
```

**What it validates:**
- ✅ Infrastructure tests via Terratest
- ✅ Security scanning via tfsec/checkov
- ✅ No hardcoded secrets

## Custom Configuration

### Option 1: Use Pre-Made Config

Copy the appropriate config to your project:

```bash
# For Node.js
cp ~/Repositories/LangChainWorkflows/examples/project_configs/nodejs_project.yaml \
   .langgraph/config/gates.yaml

# For Go
cp ~/Repositories/LangChainWorkflows/examples/project_configs/go_project.yaml \
   .langgraph/config/gates.yaml

# For Terraform
cp ~/Repositories/LangChainWorkflows/examples/project_configs/terraform_project.yaml \
   .langgraph/config/gates.yaml
```

### Option 2: Create Custom Config

Create `.langgraph/config/gates.yaml` in your project:

```yaml
extends: "~/.claude/langgraph/config/default_gates.yaml"

project:
  type: "your-type"
  language: "your-language"

  source_directories:
    - "your-src-dir/"

  test_directories:
    - "your-test-dir/"

gates:
  gate_2_coverage:
    coverage_command: "your-coverage-command"
```

## Coverage Tools by Language

### Node.js/TypeScript

**Jest:**
```bash
npm test -- --coverage
# or
jest --coverage
```

**Vitest:**
```bash
vitest run --coverage
```

**NYC (for Mocha):**
```bash
nyc mocha
```

### Go

```bash
go test -coverprofile=coverage.out -covermode=atomic ./...
go tool cover -html=coverage.out
```

### Python

```bash
pytest --cov=. --cov-report=html
```

### Terraform

**Terratest (Go-based):**
```bash
cd tests/
go test -v
```

**Terraform-compliance:**
```bash
terraform-compliance -f tests/ -p .
```

### Rust

```bash
cargo tarpaulin --out Html
```

### Java

**Maven:**
```bash
mvn clean test jacoco:report
```

**Gradle:**
```bash
./gradlew test jacocoTestReport
```

## Examples by Project Type

### 1. Node.js Express API

```bash
cd ~/projects/my-express-api
python3 ~/Repositories/LangChainWorkflows/examples/auto_detect_runner.py
```

Gate 2 will run: `jest --coverage` or `npm test -- --coverage`

### 2. Go Microservice

```bash
cd ~/projects/my-go-service
python3 ~/Repositories/LangChainWorkflows/examples/auto_detect_runner.py
```

Gate 2 will run: `go test -coverprofile=coverage.out ./...`

### 3. Terraform Infrastructure

```bash
cd ~/projects/my-infra
python3 ~/Repositories/LangChainWorkflows/examples/auto_detect_runner.py
```

Gate 2 will run: `cd tests && go test -v` (if using Terratest)
Gate 5 will run: `tfsec . && checkov -d .`

## Troubleshooting

### "No test directories detected"

**Solution:** Create a tests directory:
```bash
mkdir tests
# Add at least one test file
```

### "Coverage tool not found"

**Node.js:**
```bash
npm install --save-dev jest @types/jest
# or
npm install --save-dev vitest @vitest/coverage-v8
```

**Go:**
```bash
# Coverage is built-in, no install needed
go test -cover ./...
```

**Python:**
```bash
pip install pytest pytest-cov
```

### "Project type not detected"

The auto-detect runner will use Python defaults. You can:

1. **Create a custom config** in `.langgraph/config/gates.yaml`
2. **Add project indicator files** (e.g., `package.json` for Node)
3. **Manually specify** project type in initial_state

## Manual Project Type Override

```python
from prp_langgraph.workflows.base_prp_workflow import BasePRPWorkflow

workflow = BasePRPWorkflow()

result = workflow.execute(
    prp_file="prp/feature.md",
    initial_state={
        "project_path": ".",
        "project_type": "nodejs",  # Override auto-detection
        "source_directories": ["src", "lib"],
        "test_directories": ["tests"]
    }
)
```

## Next Steps

1. **Test on your project:**
   ```bash
   cd /path/to/your/project
   python3 ~/Repositories/LangChainWorkflows/examples/auto_detect_runner.py
   ```

2. **Review results:** Check coverage, cost, and gate status

3. **Customize config:** Create `.langgraph/config/gates.yaml` for your needs

4. **Integrate with CI/CD:** Add to your GitHub Actions/GitLab CI

---

**The workflow adapts to YOUR project structure - no forced conventions!**
