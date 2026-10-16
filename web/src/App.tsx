import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Landing from './Landing';
import Search from './Search';
import Docs from './Docs';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<Search />} />
        <Route path="/docs" element={<Docs />} />
        <Route path="*" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  );
}