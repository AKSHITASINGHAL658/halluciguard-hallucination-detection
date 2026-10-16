export async function api(method, path, body, timeoutMs = 300000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(`/api${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return await r.json();
  } finally {
    clearTimeout(timer);
  }
}
