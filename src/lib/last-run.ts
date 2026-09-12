// A tiny main-thread breadcrumb survives a renderer restart. It contains no audio/model contents.
export interface LastRun { pending: boolean; mode: string; stage: string; backend: string; updated: string }
const key = 'beatrice-last-run-v2';
export function lastRun(): LastRun | null {
  try { return JSON.parse(sessionStorage.getItem(key) ?? 'null') as LastRun | null; } catch { return null; }
}
export function rememberRun(run: Omit<LastRun, 'updated'>) {
  try { sessionStorage.setItem(key, JSON.stringify({ ...run, updated: new Date().toISOString() })); } catch { /* Storage can be blocked. */ }
}