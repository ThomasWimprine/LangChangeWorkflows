# Lesson 00: Environment Setup

**Objective**: Get your development environment ready for LangGraph development.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git
- Text editor or IDE (VS Code, PyCharm, etc.)
- Anthropic API account

## Setup Steps

### 1. Verify Python Installation

```bash
python3 --version
# Should show Python 3.10 or higher
```

If Python is not installed, download from [python.org](https://www.python.org/downloads/).

### 2. Clone Repository

```bash
git clone <your-repository-url>
cd LangChainWorkflows
```

### 3. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Your prompt should now show (venv)
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Current minimal dependencies:
- `anthropic` - Claude API client
- `python-dotenv` - Environment variable management
- `pytest` - Testing framework (for later lessons)

### 5. Get Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Copy the key (starts with `sk-ant-api03-...`)

### 6. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your key
nano .env  # or use your preferred editor
```

Add this line to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

**Important**: Never commit `.env` to git! It's already in `.gitignore`.

### 7. Verify Setup

Create a test file `test_setup.py`:

```python
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Check API key is loaded
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found in environment")
    exit(1)

print(f"✅ API key loaded (starts with: {api_key[:20]}...)")

# Test Claude API connection
try:
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'Setup successful!' and nothing else."}]
    )
    print(f"✅ Claude API connection successful!")
    print(f"   Response: {response.content[0].text}")
except Exception as e:
    print(f"❌ Claude API connection failed: {e}")
    exit(1)

print("\n🎉 All setup complete! Ready for Lesson 01.")
```

Run it:
```bash
python test_setup.py
```

Expected output:
```
✅ API key loaded (starts with: sk-ant-api03-...)
✅ Claude API connection successful!
   Response: Setup successful!

🎉 All setup complete! Ready for Lesson 01.
```

## Troubleshooting

### "python3: command not found"
- Install Python from python.org
- On Windows, check "Add Python to PATH" during installation

### "No module named 'anthropic'"
- Make sure you're in the virtual environment (`(venv)` in prompt)
- Run `pip install -r requirements.txt` again

### "Invalid API key"
- Check your `.env` file has the correct key
- Ensure no extra spaces or quotes around the key
- Verify the key is active in console.anthropic.com

### Import errors
- Activate the virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## Next Steps

Once your setup is verified, proceed to:
- **Lesson 01**: [Hello LangGraph](../01-hello-langgraph/README.md)

## Resources

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [python-dotenv Documentation](https://pypi.org/project/python-dotenv/)

---

**Lesson Status**: ✅ Complete
**Next Lesson**: 01-hello-langgraph
