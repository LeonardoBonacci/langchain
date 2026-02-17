#!/usr/bin/env python3
"""
Jira/Zephyr Ticket Reader and Analyzer with LangChain
Reads Jira tickets or Zephyr test cases and analyzes them with LangChain + Ollama
"""

import os
import requests
from typing import List, Optional
from jira import JIRA
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


def _redact(text: str, secrets: List[str]) -> str:
    if not text:
        return text
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


class ZephyrTestLoader:
    """Load Zephyr Squad test cases and convert them to LangChain Documents"""
    
    def __init__(self, jira_url: str, access_token: str):
        """
        Initialize the Zephyr Squad loader
        
        Args:
            jira_url: Your Jira instance URL (e.g., https://yourcompany.atlassian.net)
            access_token: Your Zephyr Squad API access token (JWT)
        """
        self.base_url = jira_url.rstrip('/')
        # Zephyr Squad Cloud API base URL - using v2.8 API
        self.zephyr_api_base = "https://prod-api.zephyr4jiracloud.com/connect"
        self.access_token = access_token
        self.headers = {
            "zapiAccessKey": access_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def _extract_project_key(self, issue_key: str) -> str:
        """Extract project key from issue key (e.g., WR from WR-4299)"""
        return issue_key.split('-')[0] if '-' in issue_key else issue_key
    
    def _extract_issue_id(self, issue_key: str) -> str:
        """Extract issue number from key (e.g., 4299 from WR-4299)"""
        return issue_key.split('-')[1] if '-' in issue_key else issue_key
    
    def load_test_case(self, test_case_key: str) -> Document:
        """
        Load a Zephyr Squad test case as a LangChain Document
        
        Args:
            test_case_key: The test case key (e.g., WR-T123) or test ID
            
        Returns:
            LangChain Document containing the test case data
        """
        # Use Zephyr Squad Cloud API - get test by ID
        url = f"{self.zephyr_api_base}/public/rest/api/2.0/testcase/{test_case_key}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        test_case = response.json()
        
        # Format the content
        content_parts = [
            f"Test Case ID: {test_case.get('id', test_case_key)}",
            f"Key: {test_case.get('key', 'N/A')}",
            f"Name: {test_case.get('name', 'N/A')}",
            f"Status: {test_case.get('status', 'N/A')}",
            f"Priority: {test_case.get('priority', 'N/A')}",
            f"Created By: {test_case.get('createdBy', 'N/A')}",
            f"Created On: {test_case.get('createdOn', 'N/A')}",
            "",
        ]
        
        # Add description if available
        if test_case.get('description'):
            content_parts.append("Description:")
            content_parts.append(test_case.get('description', ''))
            content_parts.append("")
        
        # Add labels if available
        if test_case.get('labels'):
            content_parts.append(f"Labels: {', '.join(test_case.get('labels', []))}")
            content_parts.append("")
        
        # Add custom fields if available
        if test_case.get('customFields'):
            content_parts.append("Custom Fields:")
            for field_name, field_value in test_case.get('customFields', {}).items():
                content_parts.append(f"  {field_name}: {field_value}")
            content_parts.append("")
        
        content = "\n".join(content_parts)
        
        # Create metadata
        metadata = {
            "test_case_id": str(test_case.get('id', test_case_key)),
            "test_case_key": test_case.get('key', 'N/A'),
            "name": test_case.get('name', 'N/A'),
            "status": test_case.get('status', 'N/A'),
            "priority": test_case.get('priority', 'N/A'),
            "type": "zephyr_squad_test_case",
            "project_key": test_case.get('projectKey', 'N/A'),
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def load_issue_tests(self, issue_key: str) -> List[Document]:
        """
        Load all test cases associated with a Jira issue using executions
        
        Args:
            issue_key: The Jira issue key (e.g., WR-4299)
            
        Returns:
            List of LangChain Documents for each test execution
        """
        # Try multiple API endpoints based on Zephyr Squad documentation
        endpoints = [
            f"{self.zephyr_api_base}/public/rest/api/2.0/executions/search/issue/{issue_key}",
            f"{self.zephyr_api_base}/public/rest/api/1.0/executions/search/issue/{issue_key}",
        ]
        
        data = None
        last_error = None
        
        for url in endpoints:
            try:
                print(f"Trying endpoint: {url}")
                response = requests.get(url, headers=self.headers)
                
                print(f"Response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    break
                else:
                    debug_http = os.getenv('DEBUG_HTTP', 'false').lower() == 'true'
                    safe_text = _redact(response.text or "", [self.access_token])
                    if debug_http:
                        print(f"Response text (redacted): {safe_text[:2000]}")
                    last_error = safe_text[:2000]
            except Exception as e:
                safe_error = _redact(str(e), [self.access_token])
                print(f"Error with endpoint: {safe_error}")
                last_error = safe_error
                continue
        
        if not data:
            raise Exception(f"Could not fetch data from any Zephyr endpoint. Last error: {last_error}")
        
        executions = data.get('searchResults', [])
        print(f"Found {len(executions)} execution(s)")
        
        documents = []
        for execution in executions:
            content_parts = [
                f"Execution ID: {execution.get('id')}",
                f"Issue Key: {execution.get('issueKey', issue_key)}",
                f"Test Case Key: {execution.get('testCaseKey', 'N/A')}",
                f"Summary: {execution.get('summary', 'N/A')}",
                f"Status: {execution.get('executionStatus', 'N/A')}",
                f"Executed By: {execution.get('executedBy', 'N/A')}",
                f"Execution Date: {execution.get('executedOn', 'N/A')}",
                f"Cycle Name: {execution.get('cycleName', 'N/A')}",
                f"Version: {execution.get('versionName', 'N/A')}",
                "",
            ]
            
            if execution.get('comment'):
                content_parts.append("Comment:")
                content_parts.append(execution.get('comment'))
                content_parts.append("")
            
            content = "\n".join(content_parts)
            
            metadata = {
                "execution_id": str(execution.get('id')),
                "issue_key": issue_key,
                "test_case_key": execution.get('testCaseKey', 'N/A'),
                "status": execution.get('executionStatus', 'N/A'),
                "type": "zephyr_squad_execution",
                "cycle_name": execution.get('cycleName', 'N/A'),
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents
        
        return documents
    
    def get_executions_for_issue(self, issue_key: str) -> List[Document]:
        """
        Load test executions for a specific issue (same as load_issue_tests for Zephyr Squad)
        
        Args:
            issue_key: The Jira issue key (e.g., WR-4299)
            
        Returns:
            List of LangChain Documents for each execution
        """
        return self.load_issue_tests(issue_key)
        
        return documents
        
        return documents


class JiraTicketLoader:
    """Load Jira tickets and convert them to LangChain Documents"""
    
    def __init__(self, jira_url: str, email: str, api_token: str):
        """
        Initialize the Jira loader
        
        Args:
            jira_url: Your Jira instance URL (e.g., https://yourcompany.atlassian.net)
            email: Your Jira account email
            api_token: Your Jira API token
        """
        self.jira = JIRA(
            server=jira_url,
            basic_auth=(email, api_token)
        )
    
    def load_issue(self, issue_key: str) -> Document:
        """
        Load a single Jira issue as a LangChain Document
        
        Args:
            issue_key: The Jira issue key (e.g., PROJ-123)
            
        Returns:
            LangChain Document containing the issue data
        """
        issue = self.jira.issue(issue_key, expand='comments')
        
        # Format the content
        content_parts = [
            f"Issue: {issue.key}",
            f"Summary: {issue.fields.summary}",
            f"Type: {issue.fields.issuetype.name}",
            f"Status: {issue.fields.status.name}",
            f"Priority: {issue.fields.priority.name if issue.fields.priority else 'None'}",
            f"Assignee: {issue.fields.assignee.displayName if issue.fields.assignee else 'Unassigned'}",
            f"Reporter: {issue.fields.reporter.displayName}",
            f"Created: {issue.fields.created}",
            f"Updated: {issue.fields.updated}",
            "",
            "Description:",
            issue.fields.description or "No description provided",
        ]
        
        # Add comments if available
        if hasattr(issue.fields, 'comment') and issue.fields.comment.comments:
            content_parts.append("")
            content_parts.append("Comments:")
            for comment in issue.fields.comment.comments:
                content_parts.append(f"\n[{comment.author.displayName} - {comment.created}]")
                content_parts.append(comment.body)
        
        content = "\n".join(content_parts)
        
        # Create metadata
        metadata = {
            "issue_key": issue.key,
            "summary": issue.fields.summary,
            "type": issue.fields.issuetype.name,
            "status": issue.fields.status.name,
            "priority": issue.fields.priority.name if issue.fields.priority else "None",
            "assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
            "reporter": issue.fields.reporter.displayName,
            "created": issue.fields.created,
            "updated": issue.fields.updated,
            "url": f"{self.jira.client_info()}/browse/{issue.key}"
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def load_issues(self, jql: str, max_results: int = 50) -> List[Document]:
        """
        Load multiple Jira issues using JQL query
        
        Args:
            jql: JQL query string
            max_results: Maximum number of results to return
            
        Returns:
            List of LangChain Documents
        """
        issues = self.jira.search_issues(jql, maxResults=max_results, expand='comments')
        return [self.load_issue(issue.key) for issue in issues]


class JiraTicketAnalyzer:
    """Analyze Jira tickets using LangChain and Ollama"""
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """
        Initialize the ticket analyzer
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.llm = OllamaLLM(model=model_name)
    
    def summarize_ticket(self, ticket_text: str) -> str:
        """
        Generate a concise summary of a Jira ticket
        
        Args:
            ticket_text: Formatted ticket text
            
        Returns:
            Summary of the ticket
        """
        template = """
        You are a technical project manager analyzing a Jira ticket.
        Provide a concise summary of the following ticket, highlighting:
        1. The main issue or request
        2. Current status and priority
        3. Key details from description and comments
        
        Ticket Information:
        {ticket_text}
        
        Summary:
        """
        
        prompt = PromptTemplate(template=template, input_variables=["ticket_text"])
        chain = prompt | self.llm
        
        result = chain.invoke({"ticket_text": ticket_text})
        return result
    
    def extract_action_items(self, ticket_text: str) -> str:
        """
        Extract action items from a Jira ticket
        
        Args:
            ticket_text: Formatted ticket text
            
        Returns:
            List of action items
        """
        template = """
        You are a technical project manager analyzing a Jira ticket.
        Extract all action items, tasks, or next steps mentioned in the ticket.
        Format them as a numbered list.
        
        Ticket Information:
        {ticket_text}
        
        Action Items:
        """
        
        prompt = PromptTemplate(template=template, input_variables=["ticket_text"])
        chain = prompt | self.llm
        
        result = chain.invoke({"ticket_text": ticket_text})
        return result
    
    def analyze_sentiment(self, ticket_text: str) -> str:
        """
        Analyze the sentiment and urgency of a ticket
        
        Args:
            ticket_text: Formatted ticket text
            
        Returns:
            Sentiment analysis
        """
        template = """
        You are a technical project manager analyzing a Jira ticket.
        Analyze the sentiment, urgency, and tone of this ticket.
        Consider both the description and any comments.
        
        Ticket Information:
        {ticket_text}
        
        Analysis:
        """
        
        prompt = PromptTemplate(template=template, input_variables=["ticket_text"])
        chain = prompt | self.llm
        
        result = chain.invoke({"ticket_text": ticket_text})
        return result


def main():
    """Main function demonstrating usage"""
    
    # Configuration - replace with your actual values
    JIRA_URL = os.getenv('JIRA_URL', 'https://yourcompany.atlassian.net')
    JIRA_EMAIL = os.getenv('JIRA_EMAIL', 'your.email@company.com')
    JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN', 'your-api-token')
    ZEPHYR_TOKEN = os.getenv('ZEPHYR_TOKEN', '')  # Optional: Zephyr access token
    ISSUE_KEY = os.getenv('JIRA_ISSUE_KEY', 'PROJ-123')
    USE_ZEPHYR = os.getenv('USE_ZEPHYR', 'false').lower() == 'true'
    
    print(f"Fetching {'Zephyr test cases for' if USE_ZEPHYR else 'Jira ticket:'} {ISSUE_KEY}")
    print("=" * 60)
    
    try:
        documents = []
        
        if USE_ZEPHYR or (ZEPHYR_TOKEN and not JIRA_API_TOKEN.startswith('ATATT')):
            # Use Zephyr API (token looks like JWT)
            print("Using Zephyr API...")
            token = ZEPHYR_TOKEN if ZEPHYR_TOKEN else JIRA_API_TOKEN
            zephyr_loader = ZephyrTestLoader(
                jira_url=JIRA_URL,
                access_token=token
            )
            
            # Check if it's a test case key (contains -T) or issue key
            if '-T' in ISSUE_KEY:
                # Load single test case
                document = zephyr_loader.load_test_case(ISSUE_KEY)
                documents = [document]
            else:
                # Load all test executions for the issue (Zephyr Squad returns executions)
                print(f"Loading test executions for issue {ISSUE_KEY}...")
                documents = zephyr_loader.load_issue_tests(ISSUE_KEY)
                
                if not documents:
                    print(f"No test cases found for issue: {ISSUE_KEY}")
                    print("Falling back to Jira API to fetch the issue...")
                    # Fall back to Jira
                    if JIRA_EMAIL and JIRA_API_TOKEN:
                        loader = JiraTicketLoader(
                            jira_url=JIRA_URL,
                            email=JIRA_EMAIL,
                            api_token=JIRA_API_TOKEN
                        )
                        document = loader.load_issue(ISSUE_KEY)
                        documents = [document]
        else:
            # Use Jira API
            print("Using Jira API...")
            loader = JiraTicketLoader(
                jira_url=JIRA_URL,
                email=JIRA_EMAIL,
                api_token=JIRA_API_TOKEN
            )
            document = loader.load_issue(ISSUE_KEY)
            documents = [document]
        
        if not documents:
            print(f"No data found for: {ISSUE_KEY}")
            return
        
        # Display information
        for i, doc in enumerate(documents, 1):
            if len(documents) > 1:
                print(f"\n{'='*60}")
                print(f"Document {i} of {len(documents)}")
                print('='*60)
            
            print(f"\n{doc.page_content}")
            print("\n" + "=" * 60)
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
        
        print("=" * 60)
        
        # Analyze with LangChain (optional)
        use_analysis = input("\nAnalyze with AI? (y/n): ").lower() == 'y'
        
        if use_analysis:
            print("\nInitializing AI analyzer...")
            analyzer = JiraTicketAnalyzer()
            
            # Combine all documents for analysis
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            
            print("\n" + "=" * 60)
            print("AI SUMMARY")
            print("=" * 60)
            summary = analyzer.summarize_ticket(combined_text)
            print(summary)
            
            print("\n" + "=" * 60)
            print("ACTION ITEMS")
            print("=" * 60)
            action_items = analyzer.extract_action_items(combined_text)
            print(action_items)
            
            print("\n" + "=" * 60)
            print("SENTIMENT ANALYSIS")
            print("=" * 60)
            sentiment = analyzer.analyze_sentiment(combined_text)
            print(sentiment)
        
    except Exception as e:
        safe_error = _redact(str(e), [JIRA_API_TOKEN, ZEPHYR_TOKEN])
        print(f"Error: {safe_error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
