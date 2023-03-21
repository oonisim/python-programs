"""
OpenAPI Chat API module
[References]
    https://platform.openai.com/docs/api-reference
    https://github.com/openai/openai-cookbook

[Models]
Available models for the endpoint is listed in:
https://platform.openai.com/docs/models/model-endpoint-compatibility
| ENDPOINT             | MODEL NAME    (As of MAR/2023)                                                  |
|----------------------|---------------------------------------------------------------------------------|
| /v1/chat/completions | gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301 |
"""
import os
import re
from typing import (
    List,
    Dict,
)

import openai


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "text-davinci-003"
]
OPENAI_MODEL_CHAT_COMPLETIONS = "gpt-3.5-turbo"
OPENAI_MODEL_TEXT_COMPLETIONS = "text-davinci-003"


# --------------------------------------------------------------------------------
# API class
# --------------------------------------------------------------------------------
class OpenAI:
    """OpenAI API implementation class"""
    def __init__(
            self,
            path_to_api_key: str,
            model_chat_completions: str = OPENAI_MODEL_CHAT_COMPLETIONS,
            model_text_completions: str = OPENAI_MODEL_TEXT_COMPLETIONS
    ):
        openai.api_key = open(file=path_to_api_key).readline().strip()
        self._model_chat_completions: str = model_chat_completions
        self._model_text_completions: str = model_text_completions

    @property
    def model_chat_completions(self):
        return self._model_chat_completions

    @property
    def model_text_completions(self):
        return self._model_text_completions

    def get_chat_completion_by_prompt(self, prompt) -> str:
        """Send a chat completion request as a prompt
        https://platform.openai.com/docs/api-reference/completions/create

        Prompt examples using the news:
        https://www.sbs.com.au/news/article/french-government-survives-no-confidence-votes-in-pension-fight/rmgr77do1

        "Summarize the text. Must be less than 15 words. Text=<text>"
        > French government survives no-confidence motions, but faces pressure over controversial pensions reform.

        "Notable figures: <text>"
        > French President Emmanuel Macron,
        > Prime Minister Elisabeth Borne,
        > Finance Minister Bruno Le Maire,
        > and Republican MP Aurelien Pradie.

        Args:
            prompt: A task instruction to the model
        Returns: Completion
        """
        response = openai.ChatCompletion.create(
            model=self.model_chat_completions,
            messages=[
                {"role": "assistant", "content": "You are short and simple."},
                {"role": "user", "content": re.sub(r"['\"/\s]+", ' ', prompt, flags=re.MULTILINE)}
            ]
        )
        return response['choices'][0]['message']['content']

    def get_chat_completion_by_messages(
            self, messages: List[Dict[str, str]]
    ) -> str:
        """Send a chat completion request as a prompt
        https://platform.openai.com/docs/api-reference/completions/create

        Message example:
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]

        Args:
            messages:
                messages as in the expected format by API.
                See https://platform.openai.com/docs/guides/chat/introduction.
        Returns: Completion
        """
        response = openai.ChatCompletion.create(
            model=self.model_chat_completions,
            messages=messages
        )
        return response['choices'][0]['message']['content']


def run_get_chat_completion_by_prompt(prompt):
    ai = OpenAI(
        path_to_api_key=f"{os.path.expanduser('~')}/.openai/api_key"
    )
    print(ai.get_chat_completion_by_prompt(prompt=prompt))


if __name__ == "__main__":
    text = 'The French government under President Emmanuel Macron on Monday survived two no-confidence motions in parliament, but still faces intense pressure over its handling of a controversial pensions reform. Prime Minister Elisabeth Borne incensed the opposition last week by announcing the government would impose a controversial pension reform without a vote in parliament, sparking accusations of anti-democratic behaviour. Its use of an article in the constitution allowing such a move also gave the opposition the right to call motions of no confidence in the government and two such demands were filed. Advertisement READ MORE Pension reform has been imposed in France without a vote. How did it happen? The 577-seat National Assembly lower rejected a motion brought by the centrist LIOT coalition that is also supported by the left, by a margin of just nine votes, much narrower than expected. It then overwhelmingly rejected a motion brought by the far-right National Rally (RN) with just 94 votes in favour. The rejection of the motions means that the reform to raise the pensions age from 62 to 64 has now been adopted by the legislature. It still needs to be signed into law by Mr Macron and may also face legal challenges. Anthony Albanese to  "reset relationship " with France on upcoming visit  24 Jun 2022, 7:38 pm Anthony Albanese to  "reset relationship " with France on upcoming visit It far from represents the end of the biggest domestic crisis of the second mandate in office of Mr Macron, who has yet to make any public comment on the controversy. "We never went so far in building a compromise as we did with this reform," Ms Borne told parliament ahead of the vote, saying her use of the article to bypass a vote was "profoundly democratic" under France "s constitution set up by postwar leader Charles de Gaulle. Garbage piles up in Paris following strikes Garbage cans overflowing with trash on the streets as collectors go on strike in Paris, France. Garbage collectors have joined the massive strikes throughout France against pension reform plans. Source: Getty / Anadolu Agency/Anadolu Agency via Getty Images Trouble ahead A new round of strikes and protests have been called on Thursday and are expected to again bring public transport to a standstill in several areas. There has been a rolling strike by rubbish collectors in Paris, leading to unsightly and unhygienic piles of trash accumulating in the French capital. The future of Ms Borne, appointed as France "s second woman premier by Mr Macron after his election victory over the far right for a second mandate, remains in doubt after she failed to secure a parliamentary majority for the reform. READ MORE France wants to raise its retirement age by two years. Why are thousands protesting? Meanwhile, it remains unclear when Mr Macron will finally make public comments over the events, amid reports he is considering an address to the nation. Since Ms Borne invoked article 49.3 of the constitution, there have also been daily protests in Paris and other cities that have on occasion turned violent. A total of 169 people were arrested nationwide on Saturday during spontaneous protests, including one that assembled 4,000 in the capital. People in the street during clashes and protests in Paris. A demonstrator holds a red flare in the middle of a crowd gathered near a fire as several thousand demonstrators gathered at Place de la Concorde, opposite the National Assembly, in Paris on 16 March, 2023 to protest against pension reform. Source: Getty / Samuel Boivin Government insiders and observers have raised fears that France is again heading for another bout of violent anti-government protests, only a few years after the "Yellow Vest" movement shook the country from 2018-2019. In order to pass, the main multi-party no confidence motion needed support from around half the 61 MPs of the traditional right-wing party The Republicans. Even after its leadership insisted they should reject the motions, 19 renegade Republicans MPs voted in favour. "I think it "s the only way out. We need to move on to something else," said one of the Republicans who voted for the ousting of the government, Aurelien Pradie. Ejecting PM  "least risky " A survey on Sunday showed the head of state "s personal rating at its lowest level since the height of the "Yellow Vest" crisis in 2019, with only 28 per cent of respondents having a positive view of him. Mr Macron has argued that the pension changes are needed to avoid crippling deficits in the coming decades linked to France "s ageing population. "Those among us who are able will gradually need to work more to finance our social model, which is one of the most generous in the world," Finance Minister Bruno Le Maire said Sunday. Opponents of the reform say it places an unfair burden on low earners, women and people doing physically wearing jobs. Opinion polls have consistently shown that two thirds of French people oppose the changes. As for Mr Macron "s options now, replacing Ms Borne would be "the least risky and the most likely to give him new momentum," Bruno Cautres of the Centre for Political Research told AFP. Calling new elections is seen as unlikely. "When you "re in this much of a cycle of unpopularity and rejection over a major reform, it "s basically suicidal" to go to the polls, Brice Teinturier, head of the polling firm Ipsos, told AFP. A Harris Interactive survey of over 2,000 people this month suggested that the only winner from a new general election would be the far right, with all other major parties losing ground. '
    run_get_chat_completion_by_prompt("Summarize the TEXT less than 20 words. Response be less than 20 words. TEXT=" + text)
