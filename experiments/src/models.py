"""
Wrapper for different LLM APIs.
"""

####################################################################################################
# Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter


####################################################################################################


####################################################################################################
# Models
MAX_TOKENS = 4096  # Max tokens for the model



def get_model(model_name, temperature, use_rate_limiter=False):
    if use_rate_limiter:
        rate_limiter = rate_limiter = InMemoryRateLimiter(
            requests_per_second=1,  # <-- Super slow! We can only make a request once every 10 seconds!!
            check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
            max_bucket_size=10,  # Controls the maximum burst size.
        )
    else:
        rate_limiter = None

    if "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=5,  # , max_tokens=MAX_TOKENS
            rate_limiter=rate_limiter,
        )
    elif "gpt" in model_name.lower():
        return ChatOpenAI(
            model_name=model_name, temperature=temperature, max_retries=5, rate_limiter=rate_limiter
        )  # , max_tokens=MAX_TOKENS)
    elif "o1" in model_name.lower() or "o3" in model_name.lower():
        return ChatOpenAI(model_name=model_name, max_retries=5, rate_limiter=rate_limiter)
    elif "claude" in model_name.lower():
        return ChatAnthropic(
            model=model_name, temperature=temperature, max_retries=3, rate_limiter=rate_limiter
        )
    else:
        llm = init_chat_model(model_name, model_provider="together", rate_limiter=rate_limiter)
        llm.model_kwargs = {"temperature": temperature}
        return llm
