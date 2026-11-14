import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { PageHead } from '../components/ui'

// A guided tour of rag-lab: what happens from raw data to a benchmark score,
// with diagrams whose boxes link to the page where each step happens.

interface Step { title: string; sub: string; to?: string; href?: string; tone?: 'data' | 'ingest' | 'target' | 'retrieve' | 'score' }

function Box({ s }: { s: Step }) {
  const body = <><b>{s.title}</b><span>{s.sub}</span></>
  const cls = `flow-box tone-${s.tone ?? 'data'}`
  if (s.to) return <Link className={cls} to={s.to}>{body}</Link>
  if (s.href) return <a className={cls} href={s.href} target="_blank" rel="noreferrer">{body}</a>
  return <div className={cls}>{body}</div>
}

function Flow({ steps, label }: { steps: Step[]; label?: string }) {
  return (
    <div className="flow-lane">
      {label && <div className="flow-label">{label}</div>}
      <div className="flow">
        {steps.map((s, i) => <div className="flow-cell" key={s.title}>{i > 0 && <span className="flow-arrow" aria-hidden>→</span>}<Box s={s} /></div>)}
      </div>
    </div>)
}

function Section({ id, n, title, children }: { id: string; n?: number; title: string; children: ReactNode }) {
  return <section id={id} className="how-section"><h2>{n !== undefined && <span className="how-n">{n}</span>}{title}</h2>{children}</section>
}

const ext = (href: string, text: string) => <a href={href} target="_blank" rel="noreferrer">{text} ↗</a>

const TOC: [string, string][] = [['overview', 'The big picture'], ['datasets', 'Datasets'], ['targets', 'Targets'], ['ingestion', 'Ingestion pipelines'],
  ['retrieval', 'Retrieval pipelines'], ['scoring', 'Scoring'], ['benchmarks', 'EnterpriseRAG-Bench vs BEIR'], ['results', 'Reading results'],
  ['code', 'Your own code'], ['disk', 'Where things live'], ['links', 'Links']]

export default function HowItWorks() {
  return (
    <div className="how">
      <PageHead title="How it works" sub="rag-lab tests a RAG system end to end: from raw documents, through chunking, embedding and indexing, to retrieval, an answer written by Claude, and a score for both. Every box below links to the page where that step happens." />
      <nav className="how-toc">{TOC.map(([id, t]) => <a key={id} href={`#${id}`}>{t}</a>)}</nav>

      <Section id="overview" title="The big picture">
        <p>Three stages, each with its own pages. Data goes in once, ingestion pipelines turn it into searchable <i>targets</i>, and retrieval pipelines search a target, answer each benchmark question and score the result.</p>
        <div className="card flow-card">
          <Flow label="1 · Data" steps={[
            { title: 'Public dataset', sub: 'EnterpriseRAG-Bench, BEIR, or your files', to: '/datasets', tone: 'data' },
            { title: 'Download', sub: 'Python extractor → data/datasets/<name>/', to: '/datasets', tone: 'data' },
            { title: 'Load', sub: 'folder → internal document store, as a snapshot', to: '/datasets', tone: 'data' }]} />
          <Flow label="2 · Ingestion" steps={[
            { title: 'Source', sub: 'dataset · subset · snapshot', to: '/ingestion', tone: 'ingest' },
            { title: 'Transform', sub: 'optional: clean, rewrite, drop', to: '/ingestion', tone: 'ingest' },
            { title: 'Chunk', sub: 'fixed, Slack-aware, or Python', to: '/ingestion', tone: 'ingest' },
            { title: 'Embed', sub: 'optional: sentence-transformers or Python', to: '/ingestion', tone: 'ingest' },
            { title: 'Target', sub: 'vectors and/or BM25 in a store', to: '/targets', tone: 'target' }]} />
          <Flow label="3 · Retrieval & benchmark" steps={[
            { title: 'Target', sub: 'the store an ingestion run filled', to: '/targets', tone: 'target' },
            { title: 'Retrieve', sub: 'vector, BM25, hybrid, or Python', to: '/retrieval', tone: 'retrieve' },
            { title: 'Rerank', sub: 'optional cross-encoder', to: '/retrieval', tone: 'retrieve' },
            { title: 'Generate', sub: 'optional: Claude answers from the chunks', to: '/retrieval', tone: 'retrieve' },
            { title: 'Benchmark', sub: 'score every question', to: '/results', tone: 'score' }]} />
        </div>
        <p className="small muted">Ingestion and benchmark runs are background jobs (see <Link to="/jobs">Jobs</Link>), run one at a time, each recorded under <Link to="/runs">Runs</Link> with its config, code versions, timings and costs.</p>
      </Section>

      <Section id="datasets" n={1} title="Datasets: download, then load">
        <p>A dataset is a folder of raw documents plus a <code>questions.jsonl</code> with benchmark questions. Getting it ready takes two explicit steps, each a button on the <Link to="/datasets">Datasets</Link> page:</p>
        <ul>
          <li><b>Download</b> (<b>Get</b>): runs the dataset's Python extractor, which writes the raw files into <code>data/datasets/&lt;name&gt;/</code>. You can read and edit the script first. Your own data: create an empty folder and upload or copy files.</li>
          <li><b>Load</b>: reads the folder into rag-lab's internal document store (<code>data/corpus.duckdb</code>). Each document becomes one normalized row (id, source, title, text, metadata, content hash), and the questions are stored next to them.</li>
        </ul>
        <p>Every load is a <b>snapshot</b>: only new, changed or deleted documents are recorded, and older snapshots stay readable. A run records which snapshot it used, so its numbers can always be traced to exact documents. The store knows nothing about chunks or vectors, so the same loaded dataset can feed any pipeline, including future kinds (a graph-building ingestion would read the same documents).</p>
        <p>Each dataset page shows metrics (document lengths, sources, question categories, how many gold documents exist), a file explorer, the questions and the snapshots. Formats: <code>.txt</code>/<code>.md</code> files, <code>.jsonl</code> lines <code>{'{"id", "text", "title"?, "source"?}'}</code>, or the EnterpriseRAG-Bench release zip.</p>
      </Section>

      <Section id="targets" n={2} title="Targets: where chunks are stored and searched">
        <p>A <Link to="/targets">target</Link> is a named, persistent search store. Types: <b>local</b> (a file on disk: exact vectors + DuckDB BM25), <b>PostgreSQL + pgvector</b>, any <b>SQL database</b> through SQLAlchemy, <b>Qdrant</b>, <b>OpenSearch</b>, <b>Elasticsearch</b>, or a <b>Python script</b> for your own store. The servers come with <code>docker-compose.yml</code>.</p>
        <p>Each target has a version that goes up with every write, and remembers which dataset, embedder and documents it holds. That is what lets <b>append</b> mode write only new or changed documents, and what keys the benchmark cache: re-running a benchmark on an unchanged target is free.</p>
      </Section>

      <Section id="ingestion" n={3} title="Ingestion pipelines: documents → target">
        <p>An <Link to="/ingestion">ingestion pipeline</Link> is a graph you draw. Every path must be <code>source → (transform) → chunk → (embed) → target</code>; one pipeline can fan out to several targets to compare settings in one run.</p>
        <div className="card flow-card">
          <Flow steps={[
            { title: 'Load', sub: 'docs of the snapshot / subset → docs.parquet', tone: 'ingest' },
            { title: 'Transform', sub: 'per document; content hash recomputed', tone: 'ingest' },
            { title: 'Chunk', sub: 'cached per document', tone: 'ingest' },
            { title: 'Embed', sub: 'cached per chunk text', tone: 'ingest' },
            { title: 'Write', sub: 'append: only changed docs · overwrite: all', tone: 'target' }]} />
        </div>
        <ul>
          <li><b>Caching:</b> every step's output is stored under a key built from its settings, its code and everything upstream. Change only the embedder and chunking is reused; grow the dataset and only new documents are chunked and embedded.</li>
          <li><b>Streaming:</b> documents and chunks move between steps as parquet files read one batch at a time, and DuckDB is capped (1 GB by default) and spills to disk, so multi-GB datasets don't need multi-GB memory.</li>
          <li><b>Previews:</b> the chunk and transform nodes can be tried on a single document before running anything.</li>
        </ul>
      </Section>

      <Section id="retrieval" n={4} title="Retrieval pipelines: question → answer">
        <p>A <Link to="/retrieval">retrieval pipeline</Link> searches one target for each benchmark question. The questions are those of the dataset the target was filled from.</p>
        <ul>
          <li><b>Retrieve</b>: <code>vector</code> (embeds the question with the target's embedder), <code>bm25</code>, <code>hybrid</code> (both, fused with Reciprocal Rank Fusion), or a Python <code>retrieve()</code>.</li>
          <li><b>Rerank</b> (optional): a cross-encoder (default <code>ms-marco-MiniLM-L-6-v2</code>) re-scores the top chunks against the question, or your own Python <code>rerank()</code>.</li>
          <li><b>Generate</b> (optional): Claude gets the question and the top chunks (10 by default), must answer from them only, cite the documents it used, and say “not in the documents” rather than guess. Needs <code>ANTHROPIC_API_KEY</code> in <code>.env</code>; the run page shows tokens and cost.</li>
          <li><b>Benchmark</b>: how many questions to ask, and whether to have the answers judged.</li>
        </ul>
        <p>You can type a single query into a retrieval pipeline to see what it returns before running a whole benchmark.</p>
      </Section>

      <Section id="scoring" n={5} title="Scoring: two layers">
        <div className="card flow-card">
          <Flow label="Always · free" steps={[
            { title: 'Retrieved documents', sub: 'chunks folded into their documents, best chunk first', tone: 'retrieve' },
            { title: 'Compare with gold document ids', sub: 'from questions.jsonl', tone: 'score' },
            { title: 'recall@1/5/10/20 · context & answer recall', sub: 'per question → per category → overall', tone: 'score' }]} />
          <Flow label="With generate + gold answers" steps={[
            { title: 'Claude’s answer', sub: 'plus the documents it cites', tone: 'retrieve' },
            { title: 'LLM judge', sub: 'compares with the gold answer', tone: 'score' },
            { title: 'correctness · completeness', sub: 'plus the judge’s reasoning per question', tone: 'score' }]} />
        </div>
        <div className="table-wrap" style={{ marginTop: 12 }}><table>
          <thead><tr><th>metric</th><th>meaning</th></tr></thead>
          <tbody>
            <tr><td><code>recall@k</code></td><td>Share of the question’s gold documents found among the top k retrieved documents.</td></tr>
            <tr><td><code>context_recall</code></td><td>Share of gold documents that made it into the prompt sent to Claude.</td></tr>
            <tr><td><code>answer_recall</code></td><td>Share of gold documents the answer cites.</td></tr>
            <tr><td><code>extra_docs</code></td><td>Cited documents that are not gold.</td></tr>
            <tr><td><code>abstained</code></td><td>% of questions where the model said the answer is not in the documents.</td></tr>
            <tr><td><code>correctness</code></td><td>Judge: the answer states the gold answer’s key facts, with nothing contradicting or invented (right / wrong, averaged to %).</td></tr>
            <tr><td><code>completeness</code></td><td>Judge: share of the gold answer’s facts the answer states.</td></tr>
          </tbody></table></div>
        <p className="small muted">The judge is EnterpriseRAG-Bench’s own scoring code for that dataset (run unchanged, so the numbers are the benchmark’s official ones; install with <code>uv run rag-lab bench setup</code>), and rag-lab’s generic Claude judge for any other dataset whose questions have a <code>gold_answer</code>. Without generate, gold answers or an API key, judging switches off and the run says why.</p>
      </Section>

      <Section id="benchmarks" n={6} title="EnterpriseRAG-Bench vs BEIR">
        <div className="table-wrap"><table>
          <thead><tr><th /><th>EnterpriseRAG-Bench</th><th>BEIR (scifact, nfcorpus, fiqa, …)</th></tr></thead>
          <tbody>
            <tr><td>What it tests</td><td>The whole RAG system: retrieval <i>and</i> the final answer</td><td>Retrieval only</td></tr>
            <tr><td>Documents</td><td>~512k synthetic company docs (Slack, Gmail, Jira, Confluence, …)</td><td>One domain per set: science, medicine, finance, Wikipedia, …</td></tr>
            <tr><td>Gold documents</td><td>Yes (470 of 500 questions; the rest have no answer in the docs)</td><td>Yes, from BEIR’s relevance judgments</td></tr>
            <tr><td>Gold answers</td><td>Yes, all 500 → correctness and completeness</td><td>No → no answer scores</td></tr>
            <tr><td>Categories</td><td>10 question types, scored separately</td><td>One</td></tr>
            <tr><td>Quick runs</td><td>The <code>dev</code> subset: 100 questions, their gold docs + 20k others</td><td>A question limit on the benchmark node</td></tr>
          </tbody></table></div>
        <p>Use BEIR sets to compare retrieval choices (chunking, embedder, BM25 vs vector vs hybrid, reranker) across domains; use EnterpriseRAG-Bench to test the system end to end. BEIR papers report nDCG@10, which rewards rank order, while rag-lab reports recall@k, so compare rag-lab runs with each other rather than with published leaderboards. Some BEIR sets are not question answering (ArguAna pairs arguments with counter-arguments; Quora finds duplicate questions), so generated answers there are for reading, not scoring.</p>
      </Section>

      <Section id="results" n={7} title="Reading results">
        <ul>
          <li><Link to="/results">Results</Link>: the latest score of every retrieval pipeline on every target, as a matrix.</li>
          <li><Link to="/runs">Runs</Link>: every run with its status, dataset, snapshot, duration and cost. A run’s page shows each step (cached or not, time, stats), the scores per category and every question.</li>
          <li>Question page: the retrieved and reranked chunks, which were gold, Claude’s answer, its cited documents and the judge’s reasoning.</li>
          <li><Link to="/diff">Compare</Link>: two runs side by side, question by question: which questions a change fixed and which it broke.</li>
          <li><Link to="/heatmap">By category</Link>: runs × question categories for one metric, to see where a pipeline is weak.</li>
        </ul>
      </Section>

      <Section id="code" n={8} title="Your own code">
        <p>Transform, chunk, embed, retrieve, rerank, dataset extractors and targets can all be Python scripts written in the app. The editor opens full-window, checks your script against the function it must define (the Contract tab), and can ask Claude to write or fix it, showing the change as a diff. A built-in node can be turned into its equivalent script as a starting point. Scripts run on this machine with your permissions.</p>
      </Section>

      <Section id="disk" n={9} title="Where things live">
        <div className="table-wrap"><table>
          <thead><tr><th>path</th><th>what</th></tr></thead>
          <tbody>
            <tr><td><code>data/datasets/&lt;name&gt;/</code></td><td>Raw dataset files, <code>questions.jsonl</code> and <code>extract.py</code></td></tr>
            <tr><td><code>data/corpus.duckdb</code></td><td>The internal document store: loaded documents, questions, snapshots, subsets</td></tr>
            <tr><td><code>data/targets/&lt;id&gt;/</code></td><td>Each target’s manifest (version, history) and, for local targets, the index</td></tr>
            <tr><td><code>workspace/</code></td><td>Saved pipelines and targets (JSON)</td></tr>
            <tr><td><code>cache/</code></td><td>Cached step outputs, shared by all runs; safe to delete</td></tr>
            <tr><td><code>runs/&lt;run_id&gt;/</code></td><td>One folder per run: config, code versions, step log, answers, scores</td></tr>
          </tbody></table></div>
        <p className="small muted">None of this is in the git repository: data stays on the machine that downloaded it.</p>
      </Section>

      <Section id="links" title="Links">
        <ul className="how-links">
          <li>{ext('https://github.com/onyx-dot-app/EnterpriseRAG-Bench', 'EnterpriseRAG-Bench')}: the benchmark, its data release and its judge</li>
          <li>{ext('https://github.com/beir-cellar/beir', 'BEIR')}: the retrieval benchmark suite, and its {ext('https://huggingface.co/BeIR', 'Hugging Face mirror')} that the extractors download from</li>
          <li>{ext('https://www.sbert.net/', 'Sentence Transformers')}: the local embedders and cross-encoder rerankers</li>
          <li>{ext('https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf', 'Reciprocal Rank Fusion')}: how hybrid search merges BM25 and vector results</li>
          <li>{ext('https://duckdb.org/docs/', 'DuckDB')}: the document store, caches and the local target’s BM25 index</li>
          <li>{ext('https://docs.claude.com/', 'Claude API docs')}: the model behind generate, the LLM judge and the editor’s AI help</li>
          <li>In this repository: <code>README.md</code> (setup, Docker, every option) and <code>scripts/get_datasets.py</code></li>
        </ul>
      </Section>
    </div>
  )
}
