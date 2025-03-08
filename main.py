import discord
from discord.ext import commands
from discord import app_commands
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login


login(token="")


model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)



TOKEN = ''
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)


chat_history = {}


@client.event
async def on_ready():
    print(f'{client.user} is ready to chat!')
    
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!chat'):
        user_id = str(message.author.id)
        user_input = message.content[len('!chat'):]
            
        if user_id not in chat_history:
            chat_history[user_id] = []
            
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        chat_history[user_id].append(input_ids)
        chat_history[user_id] = chat_history[user_id][-3:]
        
        bot_input_ids = torch.cat(chat_history[user_id], dim=-1)
        
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
        
        try:
            response_ids = model.generate(bot_input_ids, attention_mask=attention_mask, max_length=200, pad_token_id=tokenizer.eos_token_id)
            response_text = tokenizer.decode(response_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        except Exception as e:
            response_text = "Sorry, I couldn't generate a response. Please try again later."
    
        await message.channel.send(response_text)




client.run(TOKEN)
