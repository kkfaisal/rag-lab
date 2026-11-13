import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'

const dark = () => window.matchMedia('(prefers-color-scheme: dark)').matches

// A Python editor for script nodes and extractor scripts.
export default function CodeEditor({ value, onChange, height = '320px', readOnly = false }: { value: string; onChange?: (v: string) => void; height?: string; readOnly?: boolean }) {
  return (
    <div className="code-editor">
      <CodeMirror value={value} height={height} theme={dark() ? 'dark' : 'light'} extensions={[python()]}
        basicSetup={{ foldGutter: false, highlightActiveLine: false }} onChange={onChange} readOnly={readOnly} editable={!readOnly} />
    </div>
  )
}
