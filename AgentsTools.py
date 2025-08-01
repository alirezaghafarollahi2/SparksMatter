#!/usr/bin/env python
# coding: utf-8\
from typing import Annotated, Literal, Any, Union, List, Optional, Tuple
from autogen.agentchat.group import (
    ContextVariables, ReplyResult, AgentTarget, AgentNameTarget, StayTarget,
    OnCondition, StringLLMCondition,
    OnContextCondition, ExpressionContextCondition, ContextExpression,
    RevertToUserTarget, TerminateTarget
)
from pydantic import BaseModel, Field
from autogen.coding.local_commandline_code_executor import LocalCommandLineCodeExecutor
from autogen.coding.base import CodeBlock
import json
import os

project_folder = os.environ["PROJECT_FOLDER"]


from functions_SparksMatter import *
# Toolset container
class Toolset:
    def __init__(self):
        self.tools = []

    def add(self, tool):
        self.tools.append(tool)

    def as_list(self):
        return [tool.to_dict() for tool in self.tools]

    def as_json(self, indent=2):
        return json.dumps(self.as_list(), indent=indent)

toolset = Toolset()

for tool in [download_structures_from_mp, cgcnn_regression, elastic_tensor_mlp, formation_energy_mlp, analyze_generated_structure,
             generate_crystal_unconditioned, generate_crystal_band, generate_crystal_bm, generate_crystal_cs]:
    toolset.add(tool.__doc__)
tools = json.dumps(toolset.tools)

def explain_query(query: Annotated[str, "initial query AS provided by the user without modification."],
                  query_explanation: Annotated[str, "An explaination of the query, including a breakdown of key terms"],
                  context_variables: ContextVariables) -> ReplyResult:
    '''Explain the posed query. What it asks? Include a breakdown of the key terms in the query.'''

    # List of valid agents
    context_variables["task_started"] = True
    context_variables["query"] = query
    context_variables["query_explanation"] = query_explanation
    message = query_explanation

    target_agent = AgentNameTarget('scientist_1')

    return ReplyResult(
        message=message,
        target=target_agent,
        context_variables=context_variables,
    )
    
        
def create_idea(proposal: Annotated[str, "Your detailed proposal."],
                context_variables: ContextVariables,
) -> ReplyResult:
    """
    Propose your detailed proposal for the given query.
    """

    idea_created = context_variables['idea_created']

    if not idea_created: 
        context_variables["idea_created"] = True
        context_variables["idea"] = proposal        
        message = (
            f"Proposal:\n{proposal}"
        )
    
        target_agent = AgentNameTarget('scientist_2')
    
        return ReplyResult(
            message=message,
            target=target_agent,
            context_variables=context_variables
        )
    
    else:
        message = f"Proposal is already created. Did you want to revise it?"
        target_agent = AgentNameTarget('scientist_1')
    
        return ReplyResult(
            message=message,
            target=target_agent,
            context_variables=context_variables
        )
     
        

def revise_idea(proposal: Annotated[str, "Your detailed proposal."],
                context_variables: ContextVariables,
) -> ReplyResult:
    """
    Revise the idea based on provided feedback.
    """

    context_variables["idea"] = proposal
    
    message = (
        f"Revised proposal:\n{proposal}"
    )

    target_agent = AgentNameTarget('scientist_2')

    return ReplyResult(
        message=message,
        target=target_agent,
        context_variables=context_variables
    )

def critic_idea(feedback: Annotated[str, "Detailed feedback about the idea."],
                idea_approved: Annotated[Literal["yes", "no"], "Set to 'yes' if the idea is approved and no further revision is needed; set to 'no' if further revision is required."],
                context_variables: ContextVariables,
) -> ReplyResult:
    """
    Propose your detailed feedback about the proposal.
    """

    if idea_approved=="yes":
        context_variables["idea_approved"] = True
        message = f"The idea is approved by scientist_2 agent. Proceed with planning..."
        target_agent = AgentNameTarget("manager")
    elif idea_approved=="no":
        context_variables["idea_approved"] = False
        message = f"The idea is not approved-revision is needed. Here is the feedback:\n{feedback}\nProceeding with scientist_1"
        target_agent = AgentNameTarget("scientist_1")
    
    return ReplyResult(
        message=message,
        target=target_agent,
        context_variables=context_variables,
    )


class InputsInstruction(BaseModel):
    inputs: str
    instruction: str

class PlanStep(BaseModel):
    step: int = Field(description="step number in the plan")
    task: str = Field(description='tasks to be performed by the agent with the tool')
    tool: str = Field(description='tool name. "" if not applicable')
    inputs: str = Field(description='Descrption of the tool inptus. "" if not applicable.')
    depends_on: Optional[List[int]] = Field(description="List of step numbers this step depends on")
    



def create_plan(
    rationale: Annotated[str, "A detailed scientific rationale describing how the proposed steps work together to accomplish the research goal."],
    plan: Annotated[list, "List of plan steps"],
    other_tasks: Annotated[list, 'List of critical tasks beyond the exsiting tools capabilities.'],
    context_variables: ContextVariables,
) -> ReplyResult:
    '''Crate a comprehensive plan for the given task.'''

    context_variables["plan_created"] = True
    context_variables["plan_explanation"] = rationale
    context_variables["plan"] = plan
    context_variables["other_tasks"] = other_tasks

    message = f"Explanation:\n{rationale}\n\nPlan is created by the planner:\n{plan}"

    target_agent = AgentNameTarget("critic")
    
    return ReplyResult(
        message=message,
        target=target_agent,
        context_variables=context_variables,
    )

def revise_plan(
    rationale: Annotated[str, "A detailed scientific rationale describing how the proposed steps work together to accomplish the research goal."],
    plan: Annotated[str, "Detailed plan steps"],
    other_tasks: Annotated[list, 'List of critical tasks beyond the exsiting tools capabilities.'],
    context_variables: ContextVariables,
) -> ReplyResult:
    '''Revise the plan based on the feedback from critic.'''

    context_variables["plan"] = plan
    context_variables["other_tasks"] = other_tasks
    context_variables["plan_explanation"] = rationale
    
    message = f"Explanation:\n{rationale}\n\nPlan is revised by the planner:\n{plan}"

    target_agent = AgentNameTarget("critic")
    
    return ReplyResult(
        message=message,
        target=target_agent,
        context_variables=context_variables,
    )

    
class FullCriticPlan(BaseModel):
    feedback: str = Field(description='Detailed feedback')

def critic_plan(
    feedback: Annotated[str, "Detailed feedback about the plan, including the overall strategy, plan steps, agents, tools, input parameters."],
    plan_approved: Annotated[Literal["yes", "no"], "Set to 'yes' if the plan is approved and no further revision is needed; set to 'no' if further revision is required."],
    context_variables: ContextVariables
) -> ReplyResult:
    '''Submit feedback about the plan'''

    implementation_plan = FullCriticPlan(
        feedback=feedback,
    )
    
    if plan_approved=="yes":
        context_variables["plan_approved"] = True
        message = f"The plan is approved by the critic agent. Proceed with execution..."
        target_agent = AgentNameTarget("manager")
    elif plan_approved=="no":
        context_variables["plan_approved"] = False
        message = f"The plan is not approved-revision is needed."
        target_agent = AgentNameTarget("planner")
    
    return ReplyResult(
        message=message,
        target=target_agent,
        context_variables=context_variables,
    )



    
class FullCriticPlan(BaseModel):
    feedback: str = Field(description='Detailed feedback')

def critic_refined_plan(
    feedback: Annotated[str, "Detailed feedback about the plan, including the overall strategy, plan steps, agents, tools, input parameters."],
    plan_approved: Annotated[Literal["yes", "no"], "Set to 'yes' if the plan is approved and no further revision is needed; set to 'no' if further revision is required."],
    context_variables: ContextVariables
) -> ReplyResult:
    '''Submit feedback about the plan'''

    implementation_plan = FullCriticPlan(
        feedback=feedback,
    )
    
    if plan_approved=="yes":
        context_variables["plan_approved"] = True
        message = f"The plan is approved by the critic agent. Proceed with execution..."
        target_agent = AgentNameTarget("refine_manager")
    elif plan_approved=="no":
        context_variables["plan_approved"] = False
        message = f"The plan is not approved-revision is needed."
        target_agent = AgentNameTarget("planner")
    
    return ReplyResult(
        message=message,
        target=target_agent,
        context_variables=context_variables,
    )
    

def ask_user(message: Annotated[str, "Your message to the user"],
             context_variables: ContextVariables) -> ReplyResult:
    "Ask the user if the plan is approved for execution."

    plan_created = context_variables.get('plan_created', False)
    plan = context_variables.get('plan', "")
    message = f'{message}\n\nPlan:\n{plan}'

    if plan_created==False:
        return ReplyResult(
            message='The plan is not created yet. Please route to the planner to create the plan...',
            target=AgentNameTarget('manager')    
        )
    else:
        return ReplyResult(
                message=message,
                target = RevertToUserTarget()
            )


def refine_plan(message: Annotated[str, "User's query regarding the plan refinement"],
                context_variables: ContextVariables) -> ReplyResult:
    "Use this tool if the user has asked for plan modification."

    context_variables['plan_approved'] = False
    context_variables['user_approved'] = False

    return ReplyResult(
        message=message,
        context_variables=context_variables,    
        target = AgentNameTarget('planner') 
    )

def user_approved(approved: Annotated[Literal["yes", "no"], "Set to 'yes' if the user approved the plan; set to 'no' if the user did not approve the plan."],
                  context_variables: ContextVariables) -> ReplyResult:
    "Use this tool only if the user has approved the plan."

    plan_created = context_variables.get('plan_created', False)
    
    if plan_created==False:
        return ReplyResult(
            message='The plan is not created yet. Routing to planner to create the plan...',
            target=AgentNameTarget('planner')    
        )
    else:
        if approved=="yes":
            context_variables['user_approved'] = True
            return ReplyResult(
                context_variables=context_variables,
                message='User approved the plan. Terminating..',
                target=TerminateTarget(),
                )
        else:
            context_variables['user_approved'] = False
            return ReplyResult(
                context_variables=context_variables,
                message='User has not approved the plan. Plan refinement is needed!',
                target=AgentNameTarget('manager'),
                )

def terminate_process(context_variables: ContextVariables) -> ReplyResult:
    "Terminate the process"

    user_approved = context_variables.get('user_approved', False)

    if user_approved==False:
        return ReplyResult(
            message='You can only terminate the process if the user has approved the plan.',
            target=AgentNameTarget('manager')    
        )
    else:
        return ReplyResult(
                message='Process terminated',
                target=TerminateTarget(),
            )

class PythonCode(BaseModel):
    code: str = Field(..., description="Full python code to be executed")


def python_code_block(
    code: Annotated[str, "A single block of Python code to be executed"],
    code_file_name: Annotated[str, "Name of the code file to store the executed script, e.g., 'script.py'"],
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Executes a single Python code block.
    """
    messages = ""
    full_code = '#!/usr/bin/env python3\n' + code
    python_code = PythonCode(code=full_code)

    # Execute the code
    executor = LocalCommandLineCodeExecutor(timeout=60000)
    code_block = CodeBlock(language="python", code=full_code)
    result = executor.execute_code_blocks([code_block])
    exit_code = result.exit_code
    message = result.output

    # Load or initialize context
    context_path = os.path.join(project_folder, "context_variables_data.json")
    try:
        with open(context_path, 'r') as f:
            context = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        context = {}
    
    if exit_code != 0:
        context["code_error"] = python_code.model_dump()
    else:
        # Update context
        if "code" not in context or not isinstance(context["code"], list):
            context["code"] = []
        context["code"].append(python_code.model_dump())
        context["code_error"] = ""
    
    
    # Save permanent code copy
    code_output_path = os.path.join(project_folder, code_file_name)
    os.makedirs(os.path.dirname(code_output_path), exist_ok=True)
    with open(code_output_path, 'w') as f:
        f.write(full_code)

    os.remove(result.code_file)

    # Save context
    with open(context_path, 'w') as f:
        json.dump(context, f, indent=2)

    # Build result message
    if exit_code == 0:
        messages += f'Code executed successfully with message:\n{message}\n\nProceed to the next code via `python_code_block`'
    else:
        messages += f'Code execution failed with error message: {message}\n\nCorrect and try again via `python_code_block`'

    return ReplyResult(message=messages, target=AgentNameTarget("assistant"))

def terminate_execution() -> ReplyResult:
    "Only use this tool when the entire plan has been executed."
    return ReplyResult(
        message='Execution terminated',
        target=TerminateTarget(),
    )