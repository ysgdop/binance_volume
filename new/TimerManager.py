



from apscheduler.schedulers.asyncio import AsyncIOScheduler


class TimerManager:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()

    def add_interval_job(
        self,
        func,
        *,
        hours=None,
        minutes=None,
        seconds=None,
        job_id=None,
    ):
        self.scheduler.add_job(
            func,
            trigger="interval",
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            id=job_id,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )

    def add_cron_job(
        self,
        func,
        *,
        hour="*",
        minute=0,
        second=0,
        job_id=None,
    ):
        self.scheduler.add_job(
            func,
            trigger="cron",
            hour=hour,
            minute=minute,
            second=second,
            id=job_id,
            max_instances=1,
            coalesce=True,
        )

    def start(self):
        self.scheduler.start()

    def shutdown(self):
        self.scheduler.shutdown(wait=False)