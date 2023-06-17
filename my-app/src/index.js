import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './Navbar';
import CompareSkillsPage from './CompareSkillsPage';


ReactDOM.render(
  <BrowserRouter>
    <Navbar />
    <Routes>
      <Route path="/compare_Corresponding skill catalogue" element={<CompareSkillsPage />} />
      {/* Add other routes for your application */}
    </Routes>
  </BrowserRouter>,
  document.getElementById('root')
);