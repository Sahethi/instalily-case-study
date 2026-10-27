## 1. Loom Link -

I have been using agentic tools like Muse in my daily life and also when I read more about InstaLily's Lily, I felt like this case study was trying make me achieve the goal of actually accomplishing the work rather than just answering questions like most agents do these days.

We are given 191 tools and 7 services and my initial thought was that an agent shouldn’t have to go through every single tool in order to complete the request because that would waste a significant amount of time. Most of the tasks might only require like 2-3 tools, so this would require me to narrow down the right tools and then orchestrate their actions.

---

## 2. Tool-Routing Strategy

So in the tool-routing part I implemented a 2 stage architecture which is basically using GPT 4.1 mini.

In the first stage I built an LLM router which is used to examine all the tool descriptions, so like all the 191 tools and selecting the relevant took based on the user query.

In the second stage is moving towards the agent that receives the schemas and executes the whole workflow. While developing, I noticed that my compact catalog requires around 3000 tokens But in case I was using the entire tool definition, it would consume around 43000 tokens. And this is quite a significant reduction in my catalog size.

I thought of using embedding based retrieval or even hierarchical routing but I felt like given the manageable size of the catalog, I went with a single LLM routing call that considers the user’s request and is able to identify the capabilities across all the different services without extra retrieval infrastructure.

I think this has a limitation for routing recall, in case the router misses a necessary tool, the agent is not always able to recover it through search_tools.

---

## 3. Orchestration Loop

So once the tools are selected, the agent must decide how to set them up in the pipeline and how to use them. For example, I want to summarize an email in Slack. That would require me to:

1. Find the email.
2. Understand the contents of the email.
3. Find which Slack channel I want to post this email summary to.
4. Post the message.

And in this case, I made sure that my agent has an iterative loop and I have an executor which is basically handling the invocation calls for the tools. I'm checking if the arguments that I'm passing are valid and also making sure that there is some sort of a cache or a log maintained.

### When does it stop

Usually, the agent stops after it gets a final response, or it reaches 10 steps, or there is no significant progress after 2 steps. But when it reaches the limit, it is able to produce the final response without having help from additional tools.

### How errors propagate

Whenever the tool fails, the error is sent back to the model as the result, and the agent is looking into how it can correct the arguments, or maybe try to explain the error or retry. And actual tool executions, including failures, are all logged. And if there are identical calls which have previously failed, they are not re-executed.

### parallel vs. sequential

Even if multiple tools are requested together, they are still always run sequentially by the executor because that's the pattern. Each tool takes around 0.07 ms to process, so even if you're making it parallel and concurrent, it wouldn't really provide a very good benefit.

When I was testing my code, there was a very interesting issue that I ran into. When the model found 2 matching emails but attempted to delete one, despite me giving instructions to clarify this sort of issue. In order to not do that, because deleting an email is quite a serious issue, I added an executor level hold and reconsidered a safeguard, so some sort of a guardrail. This successfully improved my runs from 0/5 to 5/5.

This also helped me relate to how InstaControl addresses such issues. I feel like a lot of enterprise agents require some sort of guardrails, and only prompt instructions are not enough.

---

## 4. What I'd do with another 8 hours

In my final implementation, 34/38 local scenarios have passed, and I have 75 passing unit tests. And my response latency is around 4 seconds.

- Given 8 more hours, Firstly, I think I would try to improve the tool discovery pattern because currently the agent still struggles to recover whenever the router misses any particular capability.
- And I think, secondly, I would try to see if I can improve the calendar event retrieval because the model sometimes adds extra words, such as "meeting", to the search query when the underlying search is on actual literal title matching. I would try to see if I can make the retrieval more flexible.

I feel like my final challenge is quite something that InstaBrain does, and by that I particularly mean retaining important context and knowledge across different workflows. I feel like my current implementation relies heavily on conversation history rather than persistent memory.
