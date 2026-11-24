import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { LayoutDashboard, MessageSquare, Database, Cpu, FileCode2, Box, Activity, SlidersHorizontal } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Chat from './pages/Chat'
import Data from './pages/Data'
import Train from './pages/Train'
import Configs from './pages/Configs'
import ConfigBuilder from './pages/ConfigBuilder'
import Models from './pages/Models'
import System from './pages/System'
import Operator from './pages/Operator'
import Evaluations from './pages/Evaluations'
import Artifacts from './pages/Artifacts'
import { useState } from 'react'
import { getAuthToken, setAuthToken } from './lib/api'

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 5000 } } })

const NAV = [
  { to: '/operator', icon: SlidersHorizontal, label: 'Agent' },
  { to: '/evaluations', icon: Activity, label: 'Evaluations' },
  { to: '/artifacts', icon: Box, label: 'Artifacts' },
  { to: '/',        icon: LayoutDashboard,   label: 'Dashboard' },
  { to: '/train',   icon: Cpu,               label: 'Train'     },
  { to: '/builder', icon: SlidersHorizontal, label: 'Builder'   },
  { to: '/chat',    icon: MessageSquare,     label: 'Chat'      },
  { to: '/data',    icon: Database,          label: 'Data'      },
  { to: '/configs', icon: FileCode2,         label: 'Configs'   },
  { to: '/models',  icon: Box,               label: 'Models'    },
  { to: '/system',  icon: Activity,          label: 'System'    },
]

function Sidebar() {
  const loc = useLocation()
  const [showAccess, setShowAccess] = useState(false)
  const [token, setToken] = useState(getAuthToken)
  return (
    <aside className="studio-sidebar" style={{
      width: 200, minHeight: '100dvh',
      background: 'var(--color-surface)',
      borderRight: '1px solid var(--color-line)',
      display: 'flex', flexDirection: 'column',
      position: 'fixed', top: 0, left: 0, zIndex: 50,
    }}>
      {/* Logo */}
      <div style={{ padding: '16px 14px 14px', borderBottom: '1px solid var(--color-line)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 7, flexShrink: 0,
            background: 'var(--color-accent)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 13, fontWeight: 800, color: 'var(--color-accent-ink)',
            letterSpacing: '-0.5px',
          }}>L</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, letterSpacing: '-0.3px' }}>Latticekin</div>
            <div style={{ fontSize: 11, color: 'var(--color-ink-3)', marginTop: 1 }}>Local Studio</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '8px 6px', display: 'flex', flexDirection: 'column', gap: 1 }}>
        {NAV.map(({ to, icon: Icon, label }) => {
          const active = to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(to)
          return (
            <NavLink key={to} to={to} style={{
              display: 'flex', alignItems: 'center', gap: 9,
              padding: '8px 10px', borderRadius: 8,
              textDecoration: 'none', fontSize: 13, fontWeight: active ? 500 : 400,
              color: active ? 'var(--color-ink)' : 'var(--color-ink-2)',
              background: active ? 'var(--color-surface-3)' : 'transparent',
              transition: 'background .12s, color .12s',
            }}>
              <Icon size={15} strokeWidth={active ? 2 : 1.75} />
              {label}
            </NavLink>
          )
        })}
      </nav>

      {/* Footer */}
      <div style={{
        padding: '10px 14px', borderTop: '1px solid var(--color-line)',
        fontSize: 11, color: 'var(--color-ink-3)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <button type="button" className="btn" onClick={() => setShowAccess(value => !value)} aria-expanded={showAccess}>Server access</button>
        <span style={{ fontFamily: 'var(--font-mono)' }}>v0.1</span>
      </div>
      {showAccess && <form onSubmit={event => { event.preventDefault(); setAuthToken(token); qc.invalidateQueries(); setShowAccess(false) }} style={{ padding: 12 }}>
        <label style={{ fontSize: 12 }}>Server access token<input className="input" type="password" autoComplete="off" value={token} onChange={event => setToken(event.target.value)} /></label>
        <button type="submit" className="btn" style={{ marginTop: 8 }}>Apply</button><small style={{ display: 'block', marginTop: 6 }}>Stored for this browser tab.</small>
      </form>}
    </aside>
  )
}

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="studio-layout" style={{ display: 'flex', minHeight: '100dvh' }}>
      <Sidebar />
      <main className="studio-main" style={{ marginLeft: 200, flex: 1, minHeight: '100dvh', overflow: 'auto' }}>
        {children}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/operator" element={<Operator />} />
            <Route path="/evaluations" element={<Evaluations />} />
            <Route path="/artifacts" element={<Artifacts />} />
            <Route path="/"        element={<Dashboard />} />
            <Route path="/train"   element={<Train />} />
            <Route path="/builder" element={<ConfigBuilder />} />
            <Route path="/chat"    element={<Chat />} />
            <Route path="/data"    element={<Data />} />
            <Route path="/configs" element={<Configs />} />
            <Route path="/models"  element={<Models />} />
            <Route path="/system"  element={<System />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
