import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from aiogram import Bot, Dispatcher, types, Router
from aiogram.types import ContentType
from aiogram import F
from PIL import Image
import numpy as np
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram import Dispatcher
from aiogram import Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram import Bot
import asyncio


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
threshold = 0.5

classes = [
    "Гусь", "Индюк", "Курица", "Петух", "Страус", "Утка", "Цыпленок"
]

bot = Bot(token=TOKEN)
dp = Dispatcher(storage=MemoryStorage())

device = torch.device("cpu")

squeezenet_distill = models.squeezenet1_1(pretrained=False)
squeezenet_distill.classifier = nn.Sequential(
  nn.Dropout(p=0.5),
  nn.Conv2d(512, len(classes), kernel_size=(1, 1), stride=(1, 1)),
  nn.ReLU(inplace=True),
  nn.AdaptiveAvgPool2d((1, 1)),
)
squeezenet_distill.load_state_dict(torch.load('squeezenet_distill.pth', map_location=device))
squeezenet_distill = squeezenet_distill.to(device)
squeezenet_distill.eval()

transform = transforms.Compose([
  transforms.Resize((256, 256)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


async def predict(image_path: str):
  """Прогнозирование класса по изображению."""
  try:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
      outputs = squeezenet_distill(input_tensor)
      probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

    max_prob = np.max(probabilities)
    predicted_class = np.argmax(probabilities)

    if max_prob < threshold:
      return "На изображении ничего не найдено (уверенность ниже порога)"
    return f"Класс: {classes[predicted_class]} (уверенность: {max_prob:.2f})"
  except Exception as e:
    return f"Ошибка при обработке изображения: {e}"


router = Router()

@router.message(F.text == "/start")
async def send_welcome(message: Message):
  """Приветствие."""
  await message.answer("Привет! Отправь мне изображение птицы, и я скажу, кто это.")


@router.message(F.photo)
async def handle_photo(message: Message):
  """Обработка фотографий."""
  photo = message.photo[-1]
  file = await bot.get_file(photo.file_id)
  file_path = file.file_path
  downloaded_file = await bot.download_file(file_path)

  image_path = "temp_image.jpg"
  with open(image_path, "wb") as f:
      f.write(downloaded_file.getvalue())

  result = await predict(image_path)

  await message.answer(result)

  os.remove(image_path)


@router.message(F.any())
async def unknown_message(message: Message):
  await message.answer("Я понимаю только изображения. Пожалуйста, отправьте фото птицы.")


async def main():
  dp.include_router(router)
  await bot.delete_webhook(drop_pending_updates=True)
  print("Бот запущен...")
  await dp.start_polling(bot)


if __name__ == "__main__":
  asyncio.run(main())