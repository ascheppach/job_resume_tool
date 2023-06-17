import React from 'react';

const Navbar = () => {
  return (
    <nav>
      <ul style={{ display: 'flex', listStyleType: 'none' }}>
        <li style={{ marginRight: '10px' }}><a href="/">Home</a></li>
        <li style={{ marginRight: '10px' }}><a href="/compare_Corresponding skill catalogue" target="_blank" rel="noopener noreferrer">Search for Skill Sets</a></li>
        <li style={{ marginRight: '10px' }}><a href="/ResumeMatching">Resume Job Matching</a></li>
        <li style={{ marginRight: '10px' }}><a href="/Resume Mining">Resume Mining</a></li>
        <li><a href="/contact">Contact</a></li>
      </ul>
    </nav>
  );
};

export default Navbar;
