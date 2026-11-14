import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import {
  addEdge, Background, Controls, Handle, Position, ReactFlow, useEdgesState, useNodesState,
  type Connection, type Edge, type Node, type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { send, type Catalog, type DatasetInfo, type Graph, type GraphKind, type GraphNode, type Target } from '../api'
import CodeEditor from './CodeEditor'
import ParamForm from './ParamForm'

// Which stage may feed which.
const ALLOWED: Record<string, string[]> = {
  source: ['transform', 'chunk'], transform: ['chunk'], chunk: ['embed', 'target'], embed: ['target'],
  target: ['retrieve'], retrieve: ['rerank', 'generate', 'benchmark'], rerank: ['generate', 'benchmark'], generate: ['benchmark'],
}
const DEFAULT_IMPL: Record<string, string> = {
  source: 'dataset', transform: 'clean', chunk: 'fixed', embed: 'sentence_transformers', target: 'target',
  retrieve: 'hybrid', rerank: 'cross_encoder', generate: 'llm', benchmark: 'benchmark',
}
const HELP: Record<string, string> = {
  source: 'Documents of a dataset', transform: 'Clean, rewrite or filter documents (optional)', chunk: 'Split documents into pieces', embed: 'Turn chunks into vectors',
  target: 'The store chunks are written to / searched in', retrieve: 'Find candidate chunks for a question',
  rerank: 'Re-score the candidates', generate: 'Answer with an LLM', benchmark: 'Score against the dataset\'s questions',
}

type RFData = { node: GraphNode; first: boolean; last: boolean; title: string; summary: string }

function StageNode({ data, selected }: NodeProps<Node<RFData>>) {
  const n = data.node
  return (
    <div className={`rf-node stage-${n.stage} ${selected ? 'selected' : ''}`}>
      {!data.first && <Handle type="target" position={Position.Left} />}
      <div className="rf-head">{n.stage}</div>
      <div className="rf-body">
        <div className="rf-impl">{data.title}</div>
        <div className="rf-summary">{data.summary || ' '}</div>
      </div>
      {!data.last && <Handle type="source" position={Position.Right} />}
    </div>
  )
}
const nodeTypes = { stage: StageNode }

interface Props {
  kind: GraphKind
  graph: Graph
  catalog: Catalog
  datasets: DatasetInfo[]
  targets: Target[]
  onChange: (g: Graph) => void
  extra?: (n: GraphNode, g: Graph) => ReactNode   // extra controls in the settings panel (previews)
}

// A node canvas (React Flow) plus a settings panel for the selected node.
// The same component edits ingestion and retrieval pipelines.
export default function GraphEditor({ kind, graph, catalog, datasets, targets, onChange, extra }: Props) {
  const stages = catalog.kinds[kind]
  const describe = useCallback((n: GraphNode): { title: string; summary: string } => {
    if (n.stage === 'source') return { title: String(n.params.dataset || 'Choose a dataset'), summary: `subset ${n.params.subset ?? 'all'} · snapshot ${n.params.snapshot ?? 'latest'}` }
    if (n.stage === 'target') {
      const t = targets.find(x => x.id === n.params.target_id)
      return { title: t?.name ?? 'Choose a target', summary: kind === 'ingestion' ? `mode: ${n.params.mode ?? 'append'}` : t ? `${t.n_chunks.toLocaleString()} chunks · v${t.version}` : '' }
    }
    if (n.stage === 'benchmark') return { title: 'Benchmark', summary: `${n.params.limit ? `${n.params.limit} questions` : 'all questions'} · judge ${n.params.judge === false ? 'off' : 'auto'}` }
    return { title: n.impl === 'python' ? 'Python script' : n.impl,
             summary: Object.entries(n.params).filter(([k]) => k !== 'code' && k !== 'system_prompt').slice(0, 2).map(([k, v]) => `${k}=${v}`).join(' · ') }
  }, [targets, kind])
  const toRF = useCallback((n: GraphNode): Node<RFData> => ({
    id: n.id, type: 'stage', position: n.position,
    data: { node: n, first: n.stage === stages[0], last: n.stage === stages[stages.length - 1], title: '', summary: '' },
  }), [stages])
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<RFData>>(graph.nodes.map(toRF))
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(graph.edges.map(e => ({ id: `${e.source}>${e.target}`, ...e })))
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Report every change upward in our own (React Flow independent) graph format.
  useEffect(() => {
    onChange({ ...graph, nodes: nodes.map(n => ({ ...n.data.node, position: n.position })), edges: edges.map(e => ({ source: e.source, target: e.target })) })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges])

  const stageOf = (id: string) => nodes.find(n => n.id === id)?.data.node.stage ?? ''
  const isValid = (c: Connection | Edge) =>
    (ALLOWED[stageOf(c.source)] ?? []).includes(stageOf(c.target)) && !edges.some(e => e.target === c.target)
  const onConnect = (c: Connection) => setEdges(es => addEdge(c, es))

  const addNode = (stage: string) => {
    const id = `${stage}-${Math.random().toString(36).slice(2, 7)}`
    // Column by stage; below everything already in that column's neighbourhood, so a new node never covers another.
    const x = stages.indexOf(stage) * 290
    const near = nodes.filter(n => Math.abs(n.position.x - x) < 260)
    const y = near.length ? Math.max(...near.map(n => n.position.y)) + 150 : 60
    const params: Record<string, unknown> =
      stage === 'source' ? { dataset: datasets[0]?.name ?? '', subset: 'all', snapshot: 'latest' }
      : stage === 'target' ? { target_id: targets[0]?.id ?? '', ...(kind === 'ingestion' ? { mode: 'append' } : {}) }
      : stage === 'benchmark' ? { limit: 20, judge: 'auto' } : {}
    setNodes(ns => [...ns, toRF({ id, stage, impl: DEFAULT_IMPL[stage], params, position: { x, y } })])
    setSelectedId(id)
  }
  const update = (id: string, patch: Partial<GraphNode>) =>
    setNodes(ns => ns.map(n => n.id === id ? { ...n, data: { ...n.data, node: { ...n.data.node, ...patch } } } : n))
  const remove = (id: string) => {
    setNodes(ns => ns.filter(n => n.id !== id))
    setEdges(es => es.filter(e => e.source !== id && e.target !== id))
    setSelectedId(null)
  }

  const shown = useMemo(() => nodes.map(n => ({ ...n, data: { ...n.data, ...describe(n.data.node) } })), [nodes, describe])
  const sel = nodes.find(n => n.id === selectedId)?.data.node
  const info = sel && catalog.stages[sel.stage]?.find(i => i.impl === sel.impl)
  const ds = sel?.stage === 'source' ? datasets.find(d => d.name === sel.params.dataset) : undefined
  const tgt = sel?.stage === 'target' ? targets.find(t => t.id === sel.params.target_id) : undefined
  const setParam = (k: string, v: unknown) => sel && update(sel.id, { params: { ...sel.params, [k]: v } })

  return (
    <div className="editor">
      <div className="canvas-card">
        <div className="palette">
          <span className="small muted">Add</span>
          {stages.map(s => <button key={s} className="sm" onClick={() => addNode(s)} title={HELP[s]}>+ {s}</button>)}
          <span className="small faint" style={{ marginLeft: 'auto' }}>Drag between the dots to connect · select a node and press Backspace to delete</span>
        </div>
        <div className="canvas">
          <ReactFlow nodes={shown} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onConnect={onConnect} isValidConnection={isValid} onNodeClick={(_, n) => setSelectedId(n.id)} onPaneClick={() => setSelectedId(null)}
            fitView fitViewOptions={{ padding: 0.15, maxZoom: 1 }} proOptions={{ hideAttribution: true }} colorMode="system">
            <Background gap={20} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      </div>

      <div className="card side">
        {!sel && <p className="muted small" style={{ margin: 0 }}>Select a node to see and edit its settings and Python.</p>}
        {sel && <>
          <div className="card-head" style={{ marginBottom: 4 }}>
            <h2 style={{ textTransform: 'capitalize' }}>{sel.stage}</h2>
            <div className="actions"><button className="sm danger" onClick={() => remove(sel.id)}>Delete</button></div>
          </div>
          <p className="muted small" style={{ margin: '0 0 8px' }}>{HELP[sel.stage]}</p>

          {(catalog.stages[sel.stage] ?? []).length > 1 && (
            <div className="field" style={{ marginBottom: 8 }}><label>Implementation</label>
              <select value={sel.impl} onChange={e => update(sel.id, { impl: e.target.value, params: {} })}>
                {catalog.stages[sel.stage].map(i => <option key={i.impl} value={i.impl}>{i.impl === 'python' ? 'python (your script)' : i.impl}</option>)}
              </select></div>)}
          {info?.doc && <p className="muted small">{info.doc}</p>}

          {sel.stage === 'source' && (
            <div className="form">
              <div className="field wide"><label>Dataset</label>
                <select value={String(sel.params.dataset ?? '')} onChange={e => update(sel.id, { params: { ...sel.params, dataset: e.target.value, subset: 'all' } })}>
                  <option value="">Choose…</option>
                  {datasets.filter(d => d.n_questions > 0).map(d => <option key={d.name}>{d.name}</option>)}
                </select>
                <span className="hint">Datasets need documents and benchmark questions. <Link to="/datasets">Manage datasets</Link></span></div>
              <div className="field"><label>Subset</label>
                <select value={String(sel.params.subset ?? 'all')} onChange={e => setParam('subset', e.target.value)}>
                  <option value="all">all ({ds?.n_questions ?? 0} questions)</option>
                  {Object.entries(ds?.subsets ?? {}).map(([k, n]) => <option key={k} value={k}>{k} ({n} questions)</option>)}
                </select></div>
              <div className="field"><label>Snapshot</label>
                <select value={String(sel.params.snapshot ?? 'latest')} onChange={e => setParam('snapshot', e.target.value === 'latest' ? 'latest' : Number(e.target.value))}>
                  <option value="latest">latest</option>
                  {(ds?.snapshots ?? []).map(s => <option key={s.snapshot} value={s.snapshot}>#{s.snapshot} ({s.n_docs.toLocaleString()} docs)</option>)}
                </select></div>
            </div>)}

          {sel.stage === 'target' && (
            <div className="form">
              <div className="field wide"><label>Target</label>
                <select value={String(sel.params.target_id ?? '')} onChange={e => setParam('target_id', e.target.value)}>
                  <option value="">Choose…</option>
                  {targets.map(t => <option key={t.id} value={t.id}>{t.name} ({t.type})</option>)}
                </select>
                <span className="hint">{tgt ? `${tgt.n_chunks.toLocaleString()} chunks from ${tgt.dataset ?? 'nothing yet'} · version ${tgt.version}. ` : ''}<Link to="/targets">Manage targets</Link></span></div>
              {kind === 'ingestion' && (
                <div className="field wide"><label>Write mode</label>
                  <select value={String(sel.params.mode ?? 'append')} onChange={e => setParam('mode', e.target.value)}>
                    <option value="append">Append — add new and changed documents only</option>
                    <option value="overwrite">Overwrite — clear the target, then write everything</option>
                  </select>
                  <span className="hint">{sel.params.mode === 'overwrite' ? 'Everything currently in the target is removed first.'
                    : 'Documents already in the target with the same content and settings are skipped; changed ones are replaced.'}</span></div>)}
              {kind === 'retrieval' && <div className="field wide"><span className="hint">Questions come from the dataset this target was written from. You can benchmark against other targets when you run.</span></div>}
            </div>)}

          {sel.stage === 'benchmark' && (
            <div className="form">
              <div className="field"><label>Questions</label>
                <input type="number" value={sel.params.limit ? Number(sel.params.limit) : ''} placeholder="all"
                  onChange={e => setParam('limit', e.target.value ? Number(e.target.value) : null)} />
                <span className="hint">Empty = every question</span></div>
              <div className="field"><label>Answer judge</label>
                <select value={sel.params.judge === false ? 'off' : 'auto'} onChange={e => setParam('judge', e.target.value === 'off' ? false : 'auto')}>
                  <option value="auto">auto</option><option value="off">off</option>
                </select>
                <span className="hint">Needs a generate node and an API key</span></div>
              <div className="field wide"><span className="hint">Always computed for free: recall@k, context recall, answer-document recall.
                {catalog.has_api_key ? '' : ' No ANTHROPIC_API_KEY is set, so LLM answers and judging are off.'}</span></div>
            </div>)}

          {info && <ParamForm key={sel.id + sel.impl} info={info} params={sel.params} onChange={p => update(sel.id, { params: p })} scriptKind={sel.stage} />}

          {info?.source && sel.impl !== 'python' && (
            <div style={{ marginTop: 10 }}>
              <div className="row" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
                <span className="small"><b>Python</b> <span className="faint">built-in · {info.source_file}</span></span>
                {info.scriptable && (
                  <button className="sm" title="Replace this node with an editable script that does the same thing"
                    onClick={() => send<Record<string, unknown>>('POST', '/api/script-equivalent', { stage: sel.stage, impl: sel.impl, params: sel.params })
                      .then(params => update(sel.id, { impl: 'python', params }))}>Edit as Python script</button>)}
              </div>
              <CodeEditor key={sel.stage + sel.impl} value={info.source} readOnly height="240px" title={`Built-in ${sel.stage} · ${info.source_file}`} />
            </div>)}
          {extra?.(sel, { ...graph, nodes: nodes.map(n => n.data.node), edges })}
        </>}
      </div>
    </div>
  )
}
