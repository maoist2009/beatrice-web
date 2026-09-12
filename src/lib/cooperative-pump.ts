/** One task at a time, scheduled by a macrotask only when input is ready. No idle Promise recursion. */
export class CooperativePump {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private task: Promise<void> | null = null;
  private stopped = false;
  private generation = 0;
  constructor(private ready: () => boolean, private step: () => Promise<void>, private failed: (error: unknown) => void) {}
  wake() {
    if (this.stopped || this.timer !== null || this.task || !this.ready()) return;
    const generation = this.generation;
    this.timer = setTimeout(() => {
      this.timer = null;
      if (this.stopped || generation !== this.generation || !this.ready()) return;
      // Assign task before step can resolve. Callback exceptions never cause an unbounded retry loop.
      this.task = Promise.resolve().then(this.step).catch(error => { this.stopped = true; this.failed(error); }).finally(() => {
        this.task = null;
        if (!this.stopped && generation === this.generation) this.wake();
      });
    }, 0);
  }
  async stop() {
    this.stopped = true; this.generation++;
    if (this.timer !== null) { clearTimeout(this.timer); this.timer = null; }
    await this.task;
  }
  get pendingTasks() { return Number(this.timer !== null) + Number(this.task !== null); }
}