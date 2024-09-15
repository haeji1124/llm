from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
import streamlit as st

st.title("OpenAI Assistant")

class EventHandler(AssistantEventHandler):
    @override
    def on_event(self, event):
        if event.event == "thread.run.requires_action":
            run_id = event.data.id
            self.message = ""
            self.message_box = st.empty()
            self.handle_requires_action(event.data, run_id)

    def handle_requires_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "research_on_ddg":
                tool_outputs.append({"tool_call_id": tool.id, "output": "ddg"})
            elif tool.function.name == "research_on_wp":
                tool_outputs.append({"tool_call_id": tool.id, "output": "wp"})

        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            for text in stream.text_deltas:
                self.message += text
                self.message_box.markdown(self.message)
                print(text, end="", flush=True)
        # Initialize recent_answer in session state
        st.session_state["recent_answer"] = self.message

def set_api_key():
    st.session_state["api_key"] = st.session_state["api_key_input"]
    st.session_state["api_key_input"] = ""

# Initialize recent_answer in session state if it doesn't exist
if "recent_answer" not in st.session_state:
    st.session_state["recent_answer"] = ""

if "api_key" in st.session_state:
    with st.sidebar:
        st.text("Valid API key!")

    if "client" in st.session_state:
        client = st.session_state["client"]
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]
    else:
        client = OpenAI(api_key=st.session_state["api_key"])
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="You help search on two tools:DuckDuckGo and Wikipedia. Return the result with tool name on the top.",
            model="gpt-4-1106-preview",
            temperature=0.1,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "research_on_ddg",
                        "description": "Get the research result for the keyword on the DuckDuckGo.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "keyword": {
                                    "type": "string",
                                    "description": "full string of user's input",
                                },
                            },
                            "required": ["keyword"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "research_on_wp",
                        "description": "Get the research result for the keyword on the Wikipedia.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "keyword": {
                                    "type": "string",
                                    "description": "full string of user's input",
                                },
                            },
                            "required": ["keyword"],
                        },
                    },
                },
            ],
        )
        thread = client.beta.threads.create()
        st.session_state["client"] = client
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    if messages:
        messages = list(messages)
        messages.reverse()
        for message in messages:
            st.chat_message(message.role).write(message.content[0].text.value)

    question = st.chat_input("Type keyword what you curious")
    if question:
        with st.chat_message("user"):
            st.write(question)
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question,
        )

        with st.chat_message("ai"):
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
            
            # Use the recent_answer from session state
            if st.session_state["recent_answer"]:
                st.download_button(
                    "save this message", st.session_state["recent_answer"]
                )

else:
    st.sidebar.text_input(
        "api key",
        key="api_key_input",
        type="password",
        on_change=set_api_key,
    )