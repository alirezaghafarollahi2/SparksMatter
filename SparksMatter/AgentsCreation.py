#!/usr/bin/env python
# coding: utf-8


from SparksMatter.AgentsSystemMessage import *
from SparksMatter.AgentsTools import *
from typing import Annotated, Literal, Any, Union, List, Optional, Tuple
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig, UpdateSystemMessage, ChatResult
from autogen.agentchat.group import (
    ContextVariables, ReplyResult, AgentTarget, AgentNameTarget, StayTarget,
    OnCondition, StringLLMCondition,
    OnContextCondition, ExpressionContextCondition, ContextExpression,
    RevertToUserTarget, TerminateTarget
)


import re
from openai import OpenAI
import json
import os.path as osp
import subprocess
client = OpenAI(organization ='')

def token_usage(response):
    dic = {"prompt tokens": response.usage.prompt_tokens,
          "completion tokens" : response.usage.completion_tokens,
          "total tokens": response.usage.total_tokens,
           "reasoning tokens": response.usage.completion_tokens_details.reasoning_tokens
          }
    return dic

#Adapted from AI-Scientist
def get_response_from_llm(
    system_message,
    prompt,
    model,
    seed=42,
    reasoning_effort="medium",
    print_debug=False,
    msg_history=None,
    temperature=0.75,
    client=client):

    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": prompt}]
    
    if model in ["gpt-4o", "gpt-4-turbo"]:
        response = client.chat.completions.create(
        seed=seed,
        model=model,
        messages=[
            {"role": "developer", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        max_completion_tokens=15000
        )
        print(token_usage(response))

    elif model in ["gpt-4.1"]:
        response = client.chat.completions.create(
            seed=seed,
            model=model,
            messages=[
                {"role": "developer", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_completion_tokens=20000
        )
        print(token_usage(response))
        
    elif model in ["o1", "o1-mini", "o3", "o3-mini", "o4-mini"]:
        response = client.chat.completions.create(
            seed=seed,
            model=model,
            reasoning_effort=reasoning_effort,
            messages=[
                {"role": "developer", "content": system_message},
                *new_msg_history,
            ],
        )
        print(token_usage(response))
        
    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


#Adapted from AI-Scientist
def get_response_from_llm_web_search(
        system_message,
        prompt,
        model,
        reasoning_effort="medium",
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        client=client):

    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": prompt}]
    
    if model in ["gpt-4o", "gpt-4-turbo"]:
        response = client.responses.create(
            model=model,
            input=[
            {
              "role": "developer",
              "content": [
                {
                    "type": "input_text",
                    "text": system_message,
                }
              ]
            },
            {
              "role": "user",
              "content": [
                  {
                      "type": "input_text",
                      "text": prompt,
                }
              ]
            }
          ],
            tools=[
                {"type": "web_search_preview"
                },
                {
          "type": "code_interpreter",
          "container": {
            "type": "auto",
            "file_ids": []
          }
        }
      ]
        )

    elif model in ["gpt-4.1", "gpt-4.1-mini"]:
        response = client.responses.create(
            model=model,
            input=[
            {
              "role": "developer",
              "content": [
                {
                    "type": "input_text",
                    "text": system_message,
                }
              ]
            },
            {
              "role": "user",
              "content": [
                  {
                      "type": "input_text",
                      "text": prompt,
                }
              ]
            }
          ],
            tools=[
                {"type": "web_search_preview"
                },
                {
          "type": "code_interpreter",
          "container": {
            "type": "auto",
            "file_ids": []
          }
        }
      ]
        )
        
    elif model in ["o1", "o1-mini", "o3", "o3-mini", "o4-mini"]:
        response = client.responses.create(
            model=model,
            input=[
            {
              "role": "developer",
              "content": [
                {
                    "type": "input_text",
                    "text": system_message,
                }
              ]
            },
            {
              "role": "user",
              "content": [
                  {
                      "type": "input_text",
                      "text": prompt,
                }
              ]
            }
          ],
              reasoning={
                  "effort":"medium",
                  "summary": "detailed"
              },
            tools=[
                {"type": "web_search_preview"
                },
                {
          "type": "code_interpreter",
          "container": {
            "type": "auto",
            "file_ids": []
          }
        }
      ]
        )
        
    #content = response.choices[0].message.content
    #new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    return response.output[-1].content[0].text



# Create the agents for the group chat
def create_group_chat_idea(project_folder, seed):
    user = ConversableAgent(name="user", human_input_mode="ALWAYS", llm_config=False,)
    
    llm_config_o4_mini = LLMConfig(api_type="openai", model="o4-mini", temperature=1, cache_seed=seed, timeout=540000,)
    llm_config_o3 = LLMConfig(api_type="openai", model="o3", temperature=1, cache_seed=seed, timeout=540000,)
    llm_config_o3_research = LLMConfig(api_type="openai", model="o3-deep-research", temperature=1, cache_seed=seed, timeout=540000,)
    llm_config_o4_mini_research = LLMConfig(api_type="openai", model="o4-mini-deep-research", temperature=1, cache_seed=seed, timeout=540000,)
    llm_config_gpt_4p1 = LLMConfig(api_type="openai", model="gpt-4.1", temperature=0, cache_seed=seed, timeout=540000, parallel_tool_calls=False)
    llm_config_gpt_4p1_mini = LLMConfig( api_type="openai", model="gpt-4.1-mini", temperature=0, cache_seed=seed, timeout=540000, parallel_tool_calls=False)
    
    def create_manager_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
        """Create the context-driven manager agent prompt."""
        path = Path(project_folder)
        path.mkdir(parents=True, exist_ok=True)
        json_path = os.path.join(project_folder, "context_variables_idea.json")
        with open(json_path, "w") as f:
            json.dump(agent.context_variables.to_dict(), f, indent=2)
        json_path = os.path.join(project_folder, "chat_history_idea.json")
        with open(json_path, "w") as f:
            json.dump(messages, f, indent=2)
        
        task_started = agent.context_variables.get("task_started")
        idea_created = agent.context_variables.get("idea_created")
        idea_approved = agent.context_variables.get("idea_approved")
        plan_created = agent.context_variables.get("plan_created")
        plan_approved = agent.context_variables.get("plan_approved")
        query = agent.context_variables.get("query", "")

        prompt = MANAGER_SYSTEM_MESSAGE.format(
            query=query,
            idea_created=idea_created,
            idea_approved=idea_approved,
            plan_created=plan_created,
            plan_approved=plan_approved,
        )
        return prompt
    
    manager = ConversableAgent(
        name="manager",
        system_message="",
        llm_config=llm_config_gpt_4p1,
        functions=[refine_plan, ask_user, user_approved],
        silent=True,
        update_agent_state_before_reply=[UpdateSystemMessage(create_manager_agent_prompt)],
    )


    def create_scientist_1_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
        """Create the context-driven scientist agent prompt."""
        json_path = os.path.join(project_folder, "context_variables_idea.json")
        with open(json_path, "w") as f:
            json.dump(agent.context_variables.to_dict(), f, indent=2)
        json_path = os.path.join(project_folder, "chat_history_idea.json")
        with open(json_path, "w") as f:
            json.dump(messages, f, indent=2)
        task_started = agent.context_variables.get("task_started")
        query = agent.context_variables.get("query")
        query_explanation = agent.context_variables.get("query_explanation")
        idea_created = agent.context_variables.get("idea_created", False)
        idea_improved = agent.context_variables.get("idea_improved", False)
        idea = agent.context_variables.get("idea", "")
        tools = agent.context_variables.get("tools", "")

        if not task_started:
            prompt = SCIENTIST_AGENT_SYSTEM_MESSAGE_1
        else:
            prompt = SCIENTIST_AGENT_SYSTEM_MESSAGE_2.format(
                query=query,
                query_explanation=query_explanation,
                tools=tools,
            )    

        return prompt
    
    scientist_1 = ConversableAgent(
        name="scientist_1",
        system_message="",
        llm_config=llm_config_o3,
        functions=[create_idea, revise_idea, explain_query],
        silent=True,
        update_agent_state_before_reply=[UpdateSystemMessage(create_scientist_1_agent_prompt)],
    )


    def create_scientist_2_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
        """Create the context-driven scientist agent prompt."""
        json_path = os.path.join(project_folder, "context_variables_idea.json")
        with open(json_path, "w") as f:
            json.dump(agent.context_variables.to_dict(), f, indent=2)
        json_path = os.path.join(project_folder, "chat_history_idea.json")
        with open(json_path, "w") as f:
            json.dump(messages, f, indent=2)
        query = agent.context_variables.get("query")
        query_explanation = agent.context_variables.get("query_explanation")
        idea_created = agent.context_variables.get("idea_created", False)
        idea_improved = agent.context_variables.get("idea_improved", False)
        idea = agent.context_variables.get("idea", "")
        tools = agent.context_variables.get("tools", "")
    
        if idea_created:
            prompt = SCIENTIST_AGENT_SYSTEM_MESSAGE_3.format(
                query=query,
                query_explanation=query_explanation,
                idea=idea,
                tools=tools,
            )  
    
        return prompt
    
    scientist_2 = ConversableAgent(
        name="scientist_2",
        system_message="",
        llm_config=llm_config_gpt_4p1,
        functions=[critic_idea],
        silent=True,
        update_agent_state_before_reply=[UpdateSystemMessage(create_scientist_2_agent_prompt)],
    )

    def create_planner_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
        """Create the context-driven planner agent prompt."""
        json_path = os.path.join(project_folder, "context_variables_idea.json")
        with open(json_path, "w") as f:
            json.dump(agent.context_variables.to_dict(), f, indent=2)
        json_path = os.path.join(project_folder, "chat_history_idea.json")
        with open(json_path, "w") as f:
            json.dump(messages, f, indent=2)
        tools = agent.context_variables.get("tools")
        idea = agent.context_variables.get("idea")
        history = ""
        prev_idea = ""
        prev_plan = ""
        
        prompt = PLANNER_SYSTEM_MESSAGE.format(
            prev_idea=prev_idea,
            prev_plan=prev_plan,
            history=history,
            idea=idea,
            tools=tools,
        )
        return prompt

    
    planner = ConversableAgent(
        name="planner",
        system_message=PLANNER_SYSTEM_MESSAGE,
        llm_config=llm_config_o3,
        functions=[create_plan, revise_plan],
        silent=True,
        update_agent_state_before_reply=[UpdateSystemMessage(create_planner_agent_prompt)],
    )

    def create_critic_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
        """Create the context-driven critic agent prompt."""
        json_path = os.path.join(project_folder, "context_variables_idea.json")
        with open(json_path, "w") as f:
            json.dump(agent.context_variables.to_dict(), f, indent=2)
        json_path = os.path.join(project_folder, "chat_history_idea.json")
        with open(json_path, "w") as f:
            json.dump(messages, f, indent=2)
        agent_tools = agent.context_variables.get("assistant_agents_tools")
        agents = agent.context_variables.get("assistant_agents")
        idea = agent.context_variables.get("idea")
    
        prompt = CRITIC_SYSTEM_MESSAGE.format(
            idea=idea,
            tools=tools,
        )
        return prompt
    
    critic = ConversableAgent(
        name="critic",
        system_message=CRITIC_SYSTEM_MESSAGE,
        llm_config=llm_config_gpt_4p1,
        functions=[critic_plan],
        silent=True,
        update_agent_state_before_reply=[UpdateSystemMessage(create_critic_agent_prompt)],
    )

    manager.handoffs.add_context_condition(
        OnContextCondition(
            target=AgentTarget(scientist_1),
            condition=ExpressionContextCondition(
                expression=ContextExpression("not ${task_started}")
            )
        )
    )
    
    manager.handoffs.add_context_condition(
        OnContextCondition(
            target=AgentTarget(scientist_1),
            condition=ExpressionContextCondition(
                expression=ContextExpression("not ${idea_created} and ${task_started}")
            )
        )
    )
    
    manager.handoffs.add_context_condition(
        OnContextCondition(
            target=AgentTarget(planner),
            condition=ExpressionContextCondition(
                expression=ContextExpression("${idea_created} and not ${plan_created} and ${task_started}")
            )
        )
    )

    
    #manager.handoffs.set_after_work(RevertToUserTarget())
    scientist_1.handoffs.set_after_work(AgentTarget(manager))
    scientist_2.handoffs.set_after_work(AgentTarget(manager))
    critic.handoffs.set_after_work(AgentTarget(manager))
    planner.handoffs.set_after_work(AgentTarget(manager))
    
    return {
        "user": user,
        "planner": planner,
        "critic": critic,
        "scientist_1": scientist_1,
        "scientist_2": scientist_2,
        "manager": manager,
    }



def create_execute_group_chat(project_folder, seed):

    llm_config_o4_mini = LLMConfig(api_type="openai", model="o4-mini", temperature=1, cache_seed=seed, timeout=540000,)
    llm_config_o3 = LLMConfig(api_type="openai", model="o3", temperature=1, cache_seed=seed, timeout=540000,)
    llm_config_o3_research = LLMConfig(api_type="openai", model="o3-deep-research", temperature=1, cache_seed=seed, timeout=540000,)
    llm_config_o4_mini_research = LLMConfig(api_type="openai", model="o4-mini-deep-research", temperature=1, cache_seed=seed, timeout=540000,)
    llm_config_gpt_4p1 = LLMConfig(api_type="openai", model="gpt-4.1", temperature=0, cache_seed=seed, timeout=540000, parallel_tool_calls=False)
    llm_config_gpt_4p1_mini = LLMConfig( api_type="openai", model="gpt-4.1-mini", temperature=0, cache_seed=seed, timeout=540000, parallel_tool_calls=False)
    
    def create_asssitant_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
        """Create the context-driven assistant mpr agent prompt."""        
        json_path = os.path.join(project_folder, f"context_variables_execution.json")
        with open(json_path, "w") as f:
            json.dump(agent.context_variables.to_dict(), f, indent=2)
    
        json_path = os.path.join(project_folder, f"chat_history_execution.json")
        with open(json_path, "w") as f:
            json.dump(messages, f, indent=2)

        query = agent.context_variables.get('query')
        tools = agent.context_variables.get("tools")
        plan = agent.context_variables.get('plan')
        # Load context
        context_path = os.path.join(project_folder, "context_variables_data.json")
        try:
            with open(context_path, 'r') as f:
                context = json.load(f)
            history = context.get("execution_history", [])
            code = context.get("code", [])
            code_error = context.get("code_error", "")
            notes = context.get("execution_notes", [])
            
        except:
            history = []
            notes = []
            code = []
            code_error = ""

        prompt = ASSISTANT_AGENT_SYSTEM_MESSAGE.format(
            plan=plan,
            query=query,
            code=code,
            code_error=code_error,
            history=history,
            notes=notes,
            project_folder=project_folder,
            tools=tools,
        ) 
        return prompt
    
    assistant = ConversableAgent(
        name="assistant",
        system_message=ASSISTANT_AGENT_SYSTEM_MESSAGE,
        llm_config=llm_config_o4_mini,
        update_agent_state_before_reply=[UpdateSystemMessage(create_asssitant_agent_prompt)],
        functions=[python_code_block, terminate_execution],
        silent=True,
    )
    return {
        "assistant": assistant,
     }


def run_Matter_idea(user_request: str, seed: int, project_folder: str) -> Tuple[ChatResult, ContextVariables]:

    agents = create_group_chat_idea(project_folder, seed)
    #with open('./photovoltaic/context_variables.json', 'r') as f:
    #    context = json.load(f)
    #context_variables = ContextVariables(context)

    #context_variables['plan_created'] = False
    #context_variables['plan_approved'] = False
    #assistant_agents['assistant_agents'] = assistant_agents
    #assistant_agents['assistant_agents_tools'] = assistant_agents_tools
    
    context_variables = ContextVariables({
        "query": user_request,
        "task_started": False,
        "idea_created": False,
        "idea_approved": False,
        "plan_created": False,
        "plan_approved": False,
        "user_approved": False,
        "idea": "",
        "plan": "",
        "tools": tools,
    })

    user_to_exclude = agents['user']  # or however you identify the user
    all_agents = [agent for agent in agents.values() if agent != user_to_exclude]

    llm_config_o4_mini_group_manger = LLMConfig(
    api_type="openai",
    model="o4-mini",
    temperature=1,
    cache_seed=42,  # change the cache_seed for different trials
    timeout=540000,
    )

    # Set up the conversation pattern
    pattern = DefaultPattern(
        initial_agent=agents["manager"],
        agents=all_agents,
        user_agent=agents['user'],
        context_variables=context_variables,
        group_manager_args={"llm_config": llm_config_o4_mini_group_manger}
    )

    project_folder = project_folder

    result, context, _ = initiate_group_chat(
    pattern=pattern,
    messages=user_request,
    max_rounds=200,
    )

    json_path = os.path.join(project_folder, "context_variables_idea.json")
    with open(json_path, "w") as f:
        json.dump(context.to_dict(), f, indent=2)

    json_path = os.path.join(project_folder, "chat_history_idea.json")
    with open(json_path, "w") as f:
        json.dump(result.chat_history, f, indent=2)

    return None

def run_Matter_experiment(seed: Annotated[int, 'random seed number'],
                          project_folder: Annotated[str, 'project folder name']) -> Tuple[ChatResult, ContextVariables]:

    path = Path(project_folder)
    json_path = os.path.join(project_folder, "context_variables_idea.json")
    with open(json_path, 'r') as f:
            context = json.load(f)
    context_variables = ContextVariables(context)
    
    agents = create_execute_group_chat(project_folder, seed)
    
    llm_config_o4_mini_group_manger = LLMConfig(
    api_type="openai",
    model="o4-mini",
    temperature=1,
    cache_seed=seed,  # change the cache_seed for different trials
    timeout=540000,
    )
    
    # Set up the conversation pattern
    pattern_execute = DefaultPattern(
        initial_agent=agents['assistant'],
        user_agent=None,
        group_after_work=agents['assistant'],
        agents=[agents['assistant']],
        context_variables=context_variables,
        group_manager_args={"llm_config": llm_config_o4_mini_group_manger}
    
    )
    
    result, context, _ = initiate_group_chat(
    pattern=pattern_execute,
    messages='',
    max_rounds=200,
    )

    return None

latex_content = r"""
\documentclass[onecolumn]{{article}}
\usepackage{{graphicx}} % Required for inserting images
\usepackage{{fancyhdr}}
\usepackage{{geometry}}
\usepackage{{wrapfig}}
\usepackage{{caption}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{authblk}}
\usepackage{{setspace}}
\usepackage[most]{{tcolorbox}}
\usepackage{{tabularx}}
\usepackage{{lipsum}} % for placeholder text

% Define a custom box style
\newtcolorbox{{takeawaybox}}{{
  colback=red!10,       % Background color
  colframe=red!60,      % Border color
  fonttitle=\bfseries,   % Title font
  coltitle=black,        % Title text color
  left=0pt, right=0pt, top=0pt, bottom=0pt, % Padding
  boxrule=0.8pt,         % Border thickness
  arc=2pt,               % Corner roundness
  width=\textwidth,
  before skip=10pt,
  after skip=10pt
}}

\fancyhead{{}}\fancyfoot{{}}
\fancyhead[C]{{In the centre of the header on all pages: \thepage}}

\begin{{document}}
\date{{}}
\author{{\texttt{{AI-generated document}}}}
\maketitle

\input{paper}

\end{{document}}
"""


def run_report(project_folder):
    path = Path(project_folder)
    json_path = os.path.join(project_folder, "context_variables_idea.json")
    with open(json_path, 'r') as f:
            context_variables_idea = json.load(f)
    json_path = os.path.join(project_folder, "context_variables_data.json")
    with open(json_path, 'r') as f:
            context_variables_experiment = json.load(f)

    thoughts = context_variables_idea.get('idea', [])
    plan = context_variables_idea.get('plan')
    results = context_variables_experiment
    query = context_variables_idea.get('query')
    tools = context_variables_idea.get('tools')
    query_explanation = context_variables_idea.get('query_explanation')
    other_steps = context_variables_idea.get('other_tasks')

    document_folder = os.path.join(project_folder, "document")
###############################################################################
    prompt_1 = writer_introduction_prompt.format(
        query=query,
        query_explanation=query_explanation,
        thoughts=thoughts,
        plan=plan,
        results=results,
        )

    msg_history_report_1 = []
        
    text_report_1, msg_history_report_1 = get_response_from_llm(
        system_message=writer_system_message,
        prompt=prompt_1,
        model='o4-mini',
        reasoning_effort='high',
        msg_history=msg_history_report_1,
        print_debug=False
    )

    match = re.search(r'<tex_START>(.*?)<tex_FINISH>', text_report_1, re.DOTALL)
    
    if match:
        code_block_introduction = match.group(1).strip()
        print('introduction generated')
    else:
        print("Introduction not generated.")
    
    # Create the document directory if it doesn't exist
    os.makedirs(document_folder, exist_ok=True)
    
    # Save the file into project_folder/document/introduction.tex
    file_path = os.path.join(document_folder, "introduction.tex")
    with open(file_path, 'w') as f:
        f.writelines(code_block_introduction)

    prompt_2 = writer_methods_prompt.format(
        tools=tools,
        query=query,
        plan=plan,
        results=results,
        introduction=code_block_introduction,
        )
    msg_history_report_1 = []
    
    text_report_1, msg_history_report_1 = get_response_from_llm(
        system_message=writer_system_message,
        prompt=prompt_2,
        model='o4-mini',
        reasoning_effort='high',
        msg_history=msg_history_report_1,
        print_debug=False
    )

    match = re.search(r'<tex_START>(.*?)<tex_FINISH>', text_report_1, re.DOTALL)
    
    if match:
        code_block_methods = match.group(1).strip()
        print('methods generated')
    else:
        print("methods not generated.")


    file_path = os.path.join(document_folder, "methods.tex")
    with open(file_path, 'w') as f:
        f.writelines(code_block_methods)

    prompt_3 = writer_results_prompt.format(
        query=query,
        query_explanation=query_explanation,
        plan=plan,
        results=results,
        other_steps=other_steps,
        introduction=code_block_introduction,
        methods=code_block_methods,
        )
    msg_history_report_1 = []
    text_report_1, msg_history_report_1 = get_response_from_llm(
        system_message=writer_system_message,
        prompt=prompt_3,
        model='o4-mini',
        reasoning_effort='high',
        msg_history=msg_history_report_1,
        print_debug=False
    )

    match = re.search(r'<tex_START>(.*?)<tex_FINISH>', text_report_1, re.DOTALL)
    
    if match:
        code_block_results = match.group(1).strip()
        print('results generated')
    else:
        print("results not generated.")

    file_path = os.path.join(document_folder, "results.tex")
    with open(file_path, 'w') as f:
        f.writelines(code_block_results)

    prompt_4 = writer_outlook_prompt.format(
        query=query,
        query_explanation=query_explanation,
        other_steps=other_steps,
        introduction=code_block_introduction,
        methods=code_block_methods,
        results=code_block_results,
        )
    msg_history_report_1 = []
    text_report_1, msg_history_report_1 = get_response_from_llm(
        system_message=writer_system_message,
        prompt=prompt_4,
        model='o4-mini',
        reasoning_effort='high',
        msg_history=msg_history_report_1,
        print_debug=False
    )

    match = re.search(r'<tex_START>(.*?)<tex_FINISH>', text_report_1, re.DOTALL)
    
    if match:
        code_block_outlook = match.group(1).strip()
        print('outlook generated')
    else:
        print("outlook not generated.")

    file_path = os.path.join(document_folder, "outlook.tex")
    with open(file_path, 'w') as f:
        f.writelines(code_block_outlook)
###############################################################################
    file_path = os.path.join(document_folder, "introduction.tex")
    with open(file_path, 'r') as f:
        code_block_introduction = f.read()
    
    file_path = os.path.join(document_folder, "methods.tex")
    with open(file_path, 'r') as f:
        code_block_methods = f.read()

    file_path = os.path.join(document_folder, "results.tex")
    with open(file_path, 'r') as f:
        code_block_results = f.read()

    file_path = os.path.join(document_folder, "outlook.tex")
    with open(file_path, 'r') as f:
        code_block_outlook = f.read()
        
    whole_document = ""
    whole_document += code_block_introduction
    whole_document += '\n'+code_block_methods
    whole_document += '\n'+code_block_results
    whole_document += '\n'+code_block_outlook
    
    prompt_refine = writer_reflection_prompt.format(
        document = whole_document,
        query=query,
        results=results
        )
    
    msg_history_report = []
    text_report, _ = get_response_from_llm(
        system_message=writer_system_message,
        prompt=prompt_refine,
        model='gpt-4.1',
        temperature=0.2,
    )
    
    match = re.search(r'<tex_START>(.*?)<tex_FINISH>', text_report, re.DOTALL)
    
    if match:
        code_block_paper = match.group(1).strip()
        print('paper generated')
    else:
        print("paper not generated.")
    
    file_path = os.path.join(document_folder, "paper.tex")
    with open(file_path, 'w') as f:
        f.writelines(code_block_paper)

    document_dir = os.path.join(project_folder, "document")
    tex_file = os.path.join(document_dir, "document.tex")

    
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(latex_content.format(paper="{"+f"paper"+"}",
                                     query=query,
                                    )
               )

    for i in range(2):
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "document.tex"],
                cwd=document_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"Run {i + 1} completed successfully.")
            # Optionally, print output:
            # print("pdflatex output:", result.stdout.decode())
            # print("pdflatex errors:", result.stderr.decode())
        except subprocess.CalledProcessError as e:
            print(f"Error compiling LaTeX on run {i + 1}:", e)
            # print(e.stdout.decode())
            # print(e.stderr.decode())
            return None
        
        if i == 0:
            time.sleep(5)  # Wait for 5 seconds before the second run
    return None