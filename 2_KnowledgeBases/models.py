import json

class ResponseError(Exception):
    def __init__(self, message):
        self.message = message

class Claude3_Haiku():
    def __init__(self,  bedrock_client, temperature=0.5, topP=0.8, topK=250, maxTokenCount=4096):
        self.bedrock = bedrock_client
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        self.model_params = {
            "maxTokenCount":maxTokenCount,
            "temperature":temperature,
            "topP":topP,
            "topK":topK
        }

    def generate_response(self, prompt):
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt} \n\nAssitant: ",
            "max_tokens_to_sample": self.model_params["maxTokenCount"],
            "temperature": self.model_params["temperature"],
            "top_p": self.model_params["topP"],
            "top_k": self.model_params["topK"]
        })

        response = self.bedrock.invoke_model(body=body, modelId=self.model_id)
        response_body =json.loads(response.get("body").read())

        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise ResponseError(f"Text generation error. Error is {finish_reason}")

        input_tokens = response_body['inputTextTokenCount']
        # collect the output tokens and the output
        output_tokens = 0   # initialize
        output = ""
        for x in response_body['results']:
            output_tokens += x['tokenCount']
            output = x['outputText']

        return output, input_tokens, output_tokens

    def converse(self, messages, model_params):
        # sys_prompt = [{
        #     "text": system_prompt
        # }]
        inference_params={
            "maxTokens":model_params["maxTokenCount"],
            "temperature": model_params["temperature"]
        }
        additional_params={
            "top_p": model_params["topP"],
            "top_k": model_params["topK"]
        }

        response = self.bedrock.converse(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig=inference_params,
            additionalModelRequestFields=additional_params
        )

        output = response["output"]["message"]
        input_tokens=response["usage"]["inputTokens"]
        output_tokens=response["usage"]["outputTokens"]

        return output, input_tokens, output_tokens

class TitanText():
    def __init__(self, bedrock_client, maxTokenCount=3072, temp=0.7, topP=0.9):
        self.bedrock = bedrock_client
        self.model_id = "amazon.titan-text-premier-v1:0"
        self.model_params = {
            "maxTokenCount":maxTokenCount,
            "stopSequences":[],
            "temperature":temp,
            "topP":topP
        }
    def generate_response(self, prompt):
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": self.model_params
        })
        response = self.bedrock.invoke_model(body=body, modelId=self.model_id)
        response_body = json.loads(response.get("body").read())

        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise ResponseError(f"Text generation error. Error is {finish_reason}")

        input_tokens = response_body['inputTextTokenCount']
        # collect the output tokens and the output
        output_tokens = 0   # initialize
        output = ""
        for x in response_body['results']:
            output_tokens += x['tokenCount']
            output = x['outputText']

        return output, input_tokens, output_tokens

class TitanEmbeddings():
    def __init__(self, bedrock_client, dimensions=1024, normalize=True, embeddingTypes=['float']):
        self.bedrock = bedrock_client
        self.model_id = "amazon.titan-embed-text-v2:0"
        self.model_params = {
            "dimensions":dimensions,
            "normalize":normalize,
            "embeddingTypes":embeddingTypes
        }

    def generate_embeddings(self, text):
        body =  {
            "inputText":text,
            "dimensions": self.model_params["dimensions"],
            "normalize": self.model_params["normalize"],
            "embeddingTypes": self.model_params["embeddingTypes"]
        }
        body = json.dumps(body)
        response = self.bedrock.invoke_model(body=body, modelId=self.model_id)
        response_body = json.loads(response.get('body').read())

        return response_body['embedding'], response_body['inputTextTokenCount']
