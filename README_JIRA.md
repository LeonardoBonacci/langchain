# Jira Ticket Reader with LangChain

Read and analyze Jira tickets using the Jira REST API and LangChain with local Ollama models.

## Features

- Fetch Jira tickets using REST API
- Parse and format ticket information (summary, description, comments, status, etc.)
- AI-powered analysis using LangChain and Ollama:
  - Ticket summarization
  - Action item extraction
  - Sentiment and urgency analysis
- No cloud API keys required (uses local Ollama)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

Pull the required model:
```bash
ollama pull qwen2.5:7b
```

### 3. Generate Jira API Token

1. Log in to your Jira account
2. Go to: https://id.atlassian.com/manage-profile/security/api-tokens
3. Click "Create API token"
4. Give it a name (e.g., "LangChain Reader")
5. Copy the generated token

### 4. Configure Environment Variables

Create a `.env` file or export these variables:

```bash
export JIRA_URL="https://yourcompany.atlassian.net"
export JIRA_EMAIL="your.email@company.com"
export JIRA_API_TOKEN="your-api-token-here"
export JIRA_ISSUE_KEY="PROJ-123"
```

## Usage

### Basic Usage

```bash
python jira_ticket_reader.py
```

### Programmatic Usage

```python
from jira_ticket_reader import JiraTicketReader, JiraTicketAnalyzer

# Initialize reader
reader = JiraTicketReader(
    jira_url="https://yourcompany.atlassian.net",
    email="your.email@company.com",
    api_token="your-api-token"
)

# Fetch and format ticket
ticket_data = reader.get_ticket("PROJ-123")
formatted_ticket = reader.format_ticket(ticket_data)
print(formatted_ticket)

# Optional: Analyze with AI
analyzer = JiraTicketAnalyzer(model_name="qwen2.5:7b")
summary = analyzer.summarize_ticket(formatted_ticket)
action_items = analyzer.extract_action_items(formatted_ticket)
sentiment = analyzer.analyze_sentiment(formatted_ticket)
```

## Parameters You Need

To use this script, you need to provide:

1. **JIRA_URL**: Your Jira instance URL
   - Format: `https://yourcompany.atlassian.net`
   - Or your on-premise Jira URL: `https://jira.yourcompany.com`

2. **JIRA_EMAIL**: Your Jira account email address
   - The email you use to log in to Jira

3. **JIRA_API_TOKEN**: Your Jira API token
   - Generate from: https://id.atlassian.com/manage-profile/security/api-tokens

4. **JIRA_ISSUE_KEY**: The ticket key you want to read
   - Format: `PROJECT-123` (e.g., `DEV-456`, `SUPPORT-789`)

## Example Output

```
Fetching Jira ticket: PROJ-123
============================================================

=== JIRA TICKET: PROJ-123 ===

Summary: Implement user authentication feature
Type: Story
Status: In Progress
Priority: High
Assignee: John Doe
Reporter: Jane Smith
Created: 2026-02-15T10:30:00.000+0000
Updated: 2026-02-17T08:15:00.000+0000

Description:
We need to implement a secure user authentication system with JWT tokens...

Comments:
[John Doe - 2026-02-16T14:20:00.000+0000]
Started working on the OAuth2 integration...

============================================================

AI SUMMARY
============================================================
This ticket requests implementation of a user authentication system...
```

## Troubleshooting

### Authentication Errors
- Verify your email and API token are correct
- Ensure your API token hasn't expired
- Check that you have access to the specified Jira project

### Model Not Found
```bash
# Pull the model if you get "model not found" error
ollama pull qwen2.5:7b
```

### Connection Issues
- Verify your Jira URL is correct (with https://)
- Check your network connection
- Ensure you're not behind a firewall blocking Jira access

## Advanced Usage

### Using Different Models

```python
# Use a different Ollama model
analyzer = JiraTicketAnalyzer(model_name="llama2")
```

### Fetching Multiple Tickets

```python
issue_keys = ["PROJ-123", "PROJ-124", "PROJ-125"]
for key in issue_keys:
    ticket_data = reader.get_ticket(key)
    formatted = reader.format_ticket(ticket_data)
    print(formatted)
```

## License

MIT
