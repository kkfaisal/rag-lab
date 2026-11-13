import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  addEdge, Background, Controls, Handle, Position, ReactFlow, useEdgesState, useNodesState,
  type Connection, type Edge, type Node, type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { Catalog, DatasetInfo, Graph, GraphKind, GraphNode } from '../api'
import ParamForm from './ParamForm'

// Which stage may feed which.
const ALLOWED: Record<string, string[]> = {
  source: ['chunk'], chunk: ['embed', 'index'], embed: ['index'],
  retrieve: ['rerank', 'generate'], rerank: ['generate'],
}
const DEFAULT_IMPL: Record<string, string> = {
  source: 'dataset', chunk: 'fixed', embed: 'sentence_transformers', index: 'duckdb',
  retrieve: 'hybrid', rerank: 'cross_encoder', generate: 'llm',
}
const HELP: Record<string, string> = {
  source: 'Which documents', chunk: 'Split into pieces', embed: 'Turn chunks into vectors', index: 'Saved, searchable result',
  retrieve: 'Find candidate chunks', rerank: 'Re-score candidates', generate: 'Answer with an LLM',
}

type RFData = { node: GraphNode; first: boolean; last: boolean; badge?: ReactNode }

function StageNode({ data, selected }: NodeProps<Node<RFData>>) {
  const n = data.node
  const summary = n.stage === 'source'
    ? `${n.params.dataset || 'pick a dataset'} · ${n.params.subset ?? 'all'}`
    : Object.entries(n.params).filter(([k]) => k !== 'code' && k !== 'system_prompt').slice(0, 2).map(([k, v]) => `${k}=${v}`).join(' · ')
  return (
    <div className={`rf-node stage-${n.stage} ${selected ? 'selected' : ''}`}>
      {!data.first && <Handle type="target" position={Position.Left} />}
      <div className="rf-stage">{n.stage}</div>
      <div className="rf-impl">{n.label || n.impl}</div>
      {n.label && <div className="small muted">{n.impl}</div>}
      <div className="small muted rf-summary">{summary}</div>
      {data.badge}
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
  onChange: (g: Graph) => void
  badge?: (n: GraphNode) => ReactNode             // e.g. build status on index nodes
  extra?: (n: GraphNode, g: Graph) => ReactNode   // extra controls in the settings panel (previews, build button)
}

// A node canvas (React Flow) plus a settings panel for the selected node.
// The same component edits ingestion pipelines and retrieval setups.
export default function GraphEditor({ kind, graph, catalog, datasets, onChange, badge, extra }: Props) {
  const stages = catalog.kinds[kind]
  const toRF = useCallback((n: GraphNode): Node<RFData> => ({
    id: n.id, type: 'stage', position: n.position,
    data: { node: n, first: n.stage === stages[0], last: n.stage === stages[stages.length - 1] },
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
    const x = stages.indexOf(stage) * 280 + 20
    const y = 40 + nodes.filter(n => n.data.node.stage === stage).length * 170
    const params: Record<string, unknown> = stage === 'source' ? { dataset: datasets[0]?.name ?? '', subset: 'all', snapshot: 'latest' } : {}
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

  const shown = useMemo(() => nodes.map(n => ({ ...n, data: { ...n.data, badge: badge?.(n.data.node) } })), [nodes, badge])
  const sel = nodes.find(n => n.id === selectedId)?.data.node
  const info = sel && catalog.stages[sel.stage]?.find(i => i.impl === sel.impl)
  const ds = sel?.stage === 'source' ? datasets.find(d => d.name === sel.params.dataset) : undefined

  return (
    <div className="editor">
      <div className="canvas-col">
        <div className="palette">
          <span className="muted small">Add node:</span>
          {stages.map(s => <button key={s} onClick={() => addNode(s)} title={HELP[s]}>+ {s}</button>)}
          <span className="muted small" style={{ marginLeft: 'auto' }}>Drag from a node's right dot to the next node's left dot to connect. Select + Backspace deletes.</span>
        </div>
        <div className="canvas">
          <ReactFlow nodes={shown} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onConnect={onConnect} isValidConnection={isValid} onNodeClick={(_, n) => setSelectedId(n.id)} onPaneClick={() => setSelectedId(null)}
            fitView fitViewOptions={{ padding: 0.2, maxZoom: 1 }} proOptions={{ hideAttribution: true }} colorMode="system">
            <Background gap={20} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      </div>

      <div className="panel side">
        {!sel && <p className="muted small">Select a node to edit its settings.</p>}
        {sel && <>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <b>{sel.stage}</b>
            <button onClick={() => remove(sel.id)}>Delete node</button>
          </div>
          <p className="muted small" style={{ margin: '2px 0 10px' }}>{HELP[sel.stage]}</p>
          <div className="form">
            {(catalog.stages[sel.stage] ?? []).length > 1 && (
              <div className="field"><label>implementation</label>
                <select value={sel.impl} onChange={e => update(sel.id, { impl: e.target.value, params: {} })}>
                  {catalog.stages[sel.stage].map(i => <option key={i.impl}>{i.impl}</option>)}
                </select></div>)}
            {sel.stage === 'index' && (
              <div className="field"><label>index name</label>
                <input value={sel.label ?? ''} placeholder="e.g. fixed-1200" onChange={e => update(sel.id, { label: e.target.value })} /></div>)}
          </div>
          {info?.doc && <p className="muted small">{info.doc}</p>}
          {sel.stage === 'source' ? (
            <div className="form">
              <div className="field"><label>dataset</label>
                <select value={String(sel.params.dataset ?? '')} onChange={e => update(sel.id, { params: { ...sel.params, dataset: e.target.value, subset: 'all' } })}>
                  <option value="">choose…</option>
                  {datasets.map(d => <option key={d.name}>{d.name}</option>)}
                </select></div>
              <div className="field"><label>subset</label>
                <select value={String(sel.params.subset ?? 'all')} onChange={e => update(sel.id, { params: { ...sel.params, subset: e.target.value } })}>
                  <option value="all">all ({ds?.n_questions ?? 0} questions)</option>
                  {Object.entries(ds?.subsets ?? {}).map(([k, n]) => <option key={k} value={k}>{k} ({n} questions)</option>)}
                </select></div>
              <div className="field"><label>snapshot</label>
                <select value={String(sel.params.snapshot ?? 'latest')}
                  onChange={e => update(sel.id, { params: { ...sel.params, snapshot: e.target.value === 'latest' ? 'latest' : Number(e.target.value) } })}>
                  <option value="latest">latest</option>
                  {(ds?.snapshots ?? []).map(s => <option key={s.snapshot} value={s.snapshot}>#{s.snapshot} ({s.n_docs.toLocaleString()} docs)</option>)}
                </select></div>
            </div>
          ) : info && <ParamForm key={sel.id + sel.impl} info={info} params={sel.params} onChange={p => update(sel.id, { params: p })} />}
          {extra?.(sel, { ...graph, nodes: nodes.map(n => n.data.node), edges })}
        </>}
      </div>
    </div>
  )
}
