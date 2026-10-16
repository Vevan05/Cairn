import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Landing from './Landing';
import Search from './Search';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<Search />} />
        <Route path="*" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  );
}