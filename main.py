from aiogram import Bot, Dispatcher
from aiogram.types import Message
import asyncio

bot = Bot(token='')
dp = Dispatcher()

@dp.message()
async def echo(message: Message):
    await message.send_copy(chat_id=message.from_user.id)

async def main():
    print('Start...')
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass