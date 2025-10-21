import asyncio
from config import load_config
from worker import IndexWorker

if __name__ == "__main__":

    config = load_config()

    worker = IndexWorker(config)

    # 使用 asyncio.run() 来启动异步的 run 方法
    asyncio.run(
        worker.run(
            job_id="2",
            job_data={"content_hash": "hash2", "file_url": "./files/aqtw.pdf", "file_type": "pdf"},
        )
    )
