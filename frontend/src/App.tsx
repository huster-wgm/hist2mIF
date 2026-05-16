import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";

type JobRow = {
  id: string;
  filename: string;
  mag: string;
  status: string;
  progress_done: number;
  progress_total: number;
  progress_label: string;
  created: string;
  updated: string;
  time_ago: string;
  error: string | null | undefined;
  thumbnail_url: string | null;
  composite_url: string | null;
};

async function fetchJobs(): Promise<JobRow[]> {
  const r = await fetch("/api/jobs/");
  if (!r.ok) throw new Error("Failed to load jobs");
  return r.json();
}

async function deleteJobApi(id: string): Promise<void> {
  const r = await fetch(`/api/jobs/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error("Delete failed");
}

async function createJob(file: File, mag: string): Promise<string> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("mag", mag);
  const r = await fetch("/api/jobs/", { method: "POST", body: fd });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || "Upload failed");
  }
  const j = await r.json();
  return j.job_id as string;
}

function IconMicroscope({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden>
      <path d="M6 21h12M9 21V10a3 3 0 0 1 6 0v11" strokeLinecap="round" />
      <path d="M8 10h8M10 7h4M7 3h10v4H7V3Z" strokeLinejoin="round" />
      <path d="M4 14h3v3H4z" strokeLinejoin="round" />
    </svg>
  );
}

function IconUpload({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="M12 16V4m0 0l4 4m-4-4L8 8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 20h16" strokeLinecap="round" />
    </svg>
  );
}

function IconZap({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" />
    </svg>
  );
}

function IconMoon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconSun({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" strokeLinecap="round" />
    </svg>
  );
}

function IconDownload({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconTrash({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconCheck({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden>
      <path d="M20 6L9 17l-5-5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ProgressCell({ done, total }: { done: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  return (
    <div className="flex min-w-[140px] items-center gap-3">
      <div className="h-2 w-full max-w-[min(10rem,40vw)] overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <div
          className="h-full rounded-full bg-emerald-500 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="whitespace-nowrap font-mono text-xs tabular-nums text-gray-600 dark:text-gray-400">
        {total > 0 ? `${done}/${total}` : "—"}
      </span>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "complete") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-800 ring-1 ring-emerald-600/10 dark:bg-emerald-950/40 dark:text-emerald-300 dark:ring-emerald-500/20">
        <IconCheck className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
        Complete
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="inline-flex items-center rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-900 ring-1 ring-amber-600/15 dark:bg-amber-950/40 dark:text-amber-200">
        Running
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="inline-flex items-center rounded-full bg-rose-50 px-3 py-1 text-xs font-medium text-rose-800 ring-1 ring-rose-600/10 dark:bg-rose-950/40 dark:text-rose-300">
        Failed
      </span>
    );
  }
  return (
    <span className="inline-flex rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-600 dark:bg-gray-800 dark:text-gray-400">
      {status}
    </span>
  );
}

export default function App() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [pollErr, setPollErr] = useState<string | null>(null);
  const [mag, setMag] = useState("20x");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionErr, setActionErr] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  const refresh = useCallback(async () => {
    try {
      setPollErr(null);
      const list = await fetchJobs();
      setJobs(list);
      setLoading(false);
    } catch (e) {
      setPollErr(e instanceof Error ? e.message : "Network error");
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const anyRunning = useMemo(() => jobs.some((j) => j.status === "running"), [jobs]);

  useEffect(() => {
    if (!anyRunning) return;
    const id = window.setInterval(() => void refresh(), 1600);
    return () => window.clearInterval(id);
  }, [anyRunning, refresh]);

  const selected = useMemo(
    () => jobs.find((j) => j.id === selectedId) ?? null,
    [jobs, selectedId],
  );

  useEffect(() => {
    if (!selectedId && jobs.length) setSelectedId(jobs[0].id);
  }, [jobs, selectedId]);

  const pickFile = (f: File | null) => {
    if (!f) return;
    const ok = f.name.toLowerCase().endsWith(".tif") || f.name.toLowerCase().endsWith(".tiff");
    if (!ok) {
      setActionErr("Please use a .tif or .tiff file.");
      return;
    }
    setActionErr(null);
    setFile(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    pickFile(f ?? null);
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setActionErr("Drag & drop a .tif file, or click to browse.");
      return;
    }
    setBusy(true);
    setActionErr(null);
    try {
      const jid = await createJob(file, mag);
      await refresh();
      setSelectedId(jid);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setActionErr(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const onDelete = async (id: string) => {
    if (!window.confirm("Remove this task and its files?")) return;
    try {
      await deleteJobApi(id);
      if (selectedId === id) setSelectedId(null);
      await refresh();
    } catch {
      setActionErr("Could not delete task.");
    }
  };

  const compositeSrc =
    selected?.status === "complete" && selected.composite_url
      ? `${selected.composite_url}?v=${encodeURIComponent(selected.updated)}`
      : null;

  return (
    <div className="min-h-screen bg-lab-surface text-lab-ink transition-colors dark:bg-gray-950 dark:text-gray-100">
      <div className="mx-auto box-border min-w-0 w-full max-w-[min(120rem,calc(100vw-2*clamp(1rem,5vw,4rem)))] px-[clamp(1rem,4vw,3rem)] py-8 md:py-10">
        {/* Header */}
        <header className="mb-8 flex items-start justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-lab-purple shadow-sm ring-1 ring-violet-700/10">
              <IconMicroscope className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white">Hist2mIF</h1>
              <p className="mt-0.5 text-sm text-gray-500 dark:text-gray-400">H&amp;E to Multiplexed Immunofluorescence</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setDark((d) => !d)}
            className="rounded-xl border border-gray-200 bg-white p-2.5 text-gray-600 shadow-sm transition hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300 dark:hover:bg-gray-800"
            title={dark ? "Light mode" : "Dark mode"}
            aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {dark ? <IconSun className="h-5 w-5" /> : <IconMoon className="h-5 w-5" />}
          </button>
        </header>

        {/* Upload card */}
        <section className="mb-8 rounded-2xl border border-gray-200/80 bg-white p-6 shadow-card dark:border-gray-800 dark:bg-gray-900">
          <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-gray-500 dark:text-gray-400">
            Upload slide
          </h2>
          <form onSubmit={(e) => void onSubmit(e)} className="mt-4">
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff"
              className="hidden"
              onChange={(ev) => pickFile(ev.target.files?.[0] ?? null)}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              className={`flex w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-14 transition ${
                dragOver
                  ? "border-lab-purple bg-lab-purple-soft/60 dark:bg-violet-950/30"
                  : "border-gray-300 bg-gray-50/50 hover:border-violet-300 hover:bg-violet-50/40 dark:border-gray-600 dark:bg-gray-800/50 dark:hover:border-violet-600"
              }`}
            >
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-lab-purple-soft text-lab-purple dark:bg-violet-950 dark:text-violet-300">
                <IconUpload className="h-6 w-6" />
              </div>
              <p className="text-center text-sm font-medium text-gray-700 dark:text-gray-300">
                Drag &amp; drop a .tif file, or{" "}
                <span className="text-lab-purple dark:text-violet-400">click to browse</span>
              </p>
              {file ? (
                <p className="mt-2 font-mono text-xs text-gray-500 dark:text-gray-400">{file.name}</p>
              ) : null}
            </button>

            <div className="mt-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <span className="text-xs font-semibold uppercase tracking-[0.12em] text-gray-500 dark:text-gray-400">
                  Magnification
                </span>
                <div className="mt-2 inline-flex rounded-lg bg-gray-100 p-1 dark:bg-gray-800">
                  {(["10x", "20x"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setMag(m)}
                      className={`rounded-md px-5 py-2 text-sm font-medium transition ${
                        mag === m
                          ? "bg-lab-purple text-white shadow-sm dark:bg-violet-600"
                          : "text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>
              <button
                type="submit"
                disabled={busy}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-lab-purple px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-lab-purple-hover disabled:cursor-not-allowed disabled:opacity-60 dark:bg-violet-600 dark:hover:bg-violet-500"
              >
                <IconZap className="h-4 w-4" />
                {busy ? "Starting…" : "Run Inference"}
              </button>
            </div>

            {actionErr ? (
              <p className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-200">
                {actionErr}
              </p>
            ) : null}
            {pollErr ? (
              <p className="mt-2 text-xs text-rose-600 dark:text-rose-400">{pollErr}</p>
            ) : null}
          </form>
        </section>

        {/* Task queue */}
        <section className="rounded-2xl border border-gray-200/80 bg-white shadow-card dark:border-gray-800 dark:bg-gray-900">
          <div className="border-b border-gray-100 px-6 py-4 dark:border-gray-800">
            <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-gray-500 dark:text-gray-400">
              Task queue
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-0 text-left text-sm max-md:min-w-[36rem]">
              <thead>
                <tr className="border-b border-gray-100 bg-gray-50/80 text-xs font-semibold uppercase tracking-wide text-gray-500 dark:border-gray-800 dark:bg-gray-800/50 dark:text-gray-400">
                  <th className="px-5 py-3 font-medium">Preview</th>
                  <th className="px-5 py-3 font-medium">File</th>
                  <th className="px-5 py-3 font-medium">Mag</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                  <th className="px-5 py-3 font-medium">Progress</th>
                  <th className="px-5 py-3 font-medium">Time</th>
                  <th className="px-5 py-3 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.length === 0 && !loading ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-16 text-center text-sm text-gray-500 dark:text-gray-400">
                      No tasks yet. Upload a slide to begin.
                    </td>
                  </tr>
                ) : loading && jobs.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-sm text-gray-500">
                      Loading…
                    </td>
                  </tr>
                ) : (
                  jobs.map((j) => {
                    const thumb = j.thumbnail_url
                      ? `${j.thumbnail_url}?v=${encodeURIComponent(j.updated)}`
                      : null;
                    const dl =
                      j.status === "complete" && j.composite_url
                        ? `${j.composite_url}?v=${encodeURIComponent(j.updated)}`
                        : null;
                    const isRowSelected = selectedId === j.id;
                    return (
                      <tr
                        key={j.id}
                        onClick={() => setSelectedId(j.id)}
                        className={`cursor-pointer border-b border-gray-100 transition last:border-0 dark:border-gray-800 ${
                          isRowSelected ? "bg-violet-50/70 dark:bg-violet-950/25" : "hover:bg-gray-50/80 dark:hover:bg-gray-800/40"
                        }`}
                      >
                        <td className="px-5 py-3">
                          <div className="relative h-16 w-16 overflow-hidden rounded-lg border border-gray-200 bg-gray-100 dark:border-gray-700 dark:bg-gray-800">
                            {thumb ? (
                              <img src={thumb} alt="" className="h-full w-full object-cover" />
                            ) : (
                              <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-900" />
                            )}
                            {j.status === "running" ? (
                              <div className="absolute inset-0 flex items-center justify-center bg-black/25">
                                <span className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                              </div>
                            ) : null}
                          </div>
                        </td>
                        <td className="max-w-[180px] truncate px-5 py-3 font-medium text-gray-900 dark:text-gray-100" title={j.filename}>
                          {j.filename}
                        </td>
                        <td className="px-5 py-3">
                          <span className="inline-flex rounded-md bg-gray-100 px-2 py-0.5 font-mono text-xs text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                            {j.mag}
                          </span>
                        </td>
                        <td className="px-5 py-3">
                          <StatusBadge status={j.status} />
                        </td>
                        <td className="px-5 py-3">
                          <ProgressCell done={j.progress_done} total={j.progress_total} />
                        </td>
                        <td className="px-5 py-3 font-mono text-xs text-gray-500 dark:text-gray-400">{j.time_ago}</td>
                        <td className="px-5 py-3">
                          <div className="flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
                            {dl ? (
                              <a
                                href={dl}
                                download
                                className="rounded-lg p-2 text-gray-500 transition hover:bg-gray-100 hover:text-lab-purple dark:hover:bg-gray-800 dark:hover:text-violet-400"
                                title="Download PNG"
                                aria-label="Download composite"
                              >
                                <IconDownload className="h-5 w-5" />
                              </a>
                            ) : (
                              <span className="p-2 text-gray-300 dark:text-gray-600">
                                <IconDownload className="h-5 w-5" />
                              </span>
                            )}
                            <button
                              type="button"
                              onClick={() => void onDelete(j.id)}
                              className="rounded-lg p-2 text-gray-500 transition hover:bg-rose-50 hover:text-rose-600 dark:hover:bg-rose-950/40 dark:hover:text-rose-400"
                              title="Delete"
                              aria-label="Delete task"
                            >
                              <IconTrash className="h-5 w-5" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Selected preview (large) — below queue, matches “preview” emphasis */}
        {selected && (
          <div className="mt-8 rounded-2xl border border-gray-200/80 bg-white p-6 shadow-card dark:border-gray-800 dark:bg-gray-900">
            <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-gray-500 dark:text-gray-400">
              Preview — {selected.filename}
            </h3>
            <div className="mt-4 flex min-h-[200px] items-center justify-center overflow-hidden rounded-xl border border-gray-100 bg-gray-50 dark:border-gray-800 dark:bg-gray-950">
              {compositeSrc ? (
                <img src={compositeSrc} alt="Virtual mIF composite" className="max-h-[420px] w-full object-contain" />
              ) : (
                <p className="px-6 py-12 text-center text-sm text-gray-500 dark:text-gray-400">
                  {selected.status === "running"
                    ? "Inference in progress…"
                    : selected.status === "failed"
                      ? selected.error ?? "Task failed."
                      : "Select a completed task to view the composite."}
                </p>
              )}
            </div>
            {selected.status === "complete" && compositeSrc ? (
              <a
                href={compositeSrc}
                download
                className="mt-4 inline-flex items-center gap-2 rounded-xl bg-lab-purple px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-lab-purple-hover dark:bg-violet-600 dark:hover:bg-violet-500"
              >
                <IconDownload className="h-4 w-4" />
                Download PNG
              </a>
            ) : null}
          </div>
        )}

        <p className="mt-8 text-center text-xs text-gray-400 dark:text-gray-600">
          Hist2mIF · GigaTIME is research-only · not for clinical use
        </p>
      </div>
    </div>
  );
}
