import React, { useState, useRef } from 'react';
import { WithContext as TagInput } from 'react-tag-input';
import axios from 'axios';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

const CompareSkillsPage = () => {
  const [skill, setSkill] = useState('');
  const [skills, setSkills] = useState([]);
  const [currentSkill, setCurrentSkill] = useState('');
  const inputRef2 = useRef(null);
  const [skillList, setSkillList] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [importantSkills, setImportantSkills] = useState([]);

  const handleSkillChange = event => {
    setSkill(event.target.value);
  };

  const handleSkillsChange = event => {
    setCurrentSkill(event.target.value);
  };

  const handleSkillsKeyDown = event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      if (currentSkill.trim() !== '') {
        setSkills(prevSkills => [...prevSkills, currentSkill]);
        setCurrentSkill('');
      }
    }
  };

  const handleTagDelete = index => {
    setSkills(prevSkills => prevSkills.filter((_, i) => i !== index));
  };

  const handleSkillSubmit = event => {
    event.preventDefault();
    // Process the skill and corresponding skill catalogue here (e.g., make API calls, update state, etc.)
    console.log('Skillcluster:', skill);
    console.log('Corresponding skill catalogue:', skills);
    setSkillList(prevSkillList => [...prevSkillList, { skill, skills }]);
    setSkill('');
    setSkills([]);
  };

  const handleFileChange = event => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
  };

  const handleSearchClick = () => {
    const formData = new FormData();

    // Append the skillList and importantSkills to the formData
    formData.append('skillList', JSON.stringify(skillList));
    formData.append('importantSkills', JSON.stringify(importantSkills));

    // Append the file contents to the formData
    selectedFiles.forEach((file, index) => {
      formData.append(`file${index}`, file);
    });

    axios
      .post('http://127.0.0.1:5000/searchApplicants', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      .then(response => {
        console.log('Applicants searched successfully');
        // Do something with the response if needed
      })
      .catch(error => {
        console.error('Error searching applicants:', error);
      });
  };

  return (
    <div>
      <h1>Step 1: Define Important Skills</h1>

      <DndProvider backend={HTML5Backend}>
        <TagInput
          tags={importantSkills}
          handleAddition={tag => setImportantSkills([...importantSkills, tag])}
          handleDelete={index => setImportantSkills(importantSkills.filter((_, i) => i !== index))}
          placeholder="Enter important skills..."
        />
      </DndProvider>

      <h1>Step 2: Upload Resumes</h1>
      <input type="file" onChange={handleFileChange} multiple />

      <h1>Step 3: Search for Skill Sets</h1>

      <form onSubmit={handleSkillSubmit}>
        <div>
          <label>
            Skillcluster:
            <input
              type="text"
              value={skill}
              onChange={handleSkillChange}
              placeholder="Enter a Skillcluster..."
            />
          </label>
        </div>
        <div>
          <label>
            Corresponding skill catalogue:
            <div className="tag-container">
              {skills.map((skill, index) => (
                <div key={index} className="tag">
                  {skill}
                  <button onClick={() => handleTagDelete(index)}>x</button>
                </div>
              ))}
            </div>
            <input
              ref={inputRef2}
              type="text"
              value={currentSkill}
              onChange={handleSkillsChange}
              onKeyDown={handleSkillsKeyDown}
              placeholder="Enter skills..."
            />
          </label>
        </div>
        <button type="submit">Submit</button>
      </form>

      <h2>Selected Skill Set:</h2>
      <ul>
        {skillList.map((item, index) => (
          <li key={index}>
            Skillcluster: {item.skill}, Corresponding Skill Catalogue: {item.skills.join(', ')}
          </li>
        ))}
      </ul>

      <div style={{ textAlign: 'center', marginTop: '20px' }}>
        <button
          onClick={handleSearchClick}
          style={{ width: '200px', height: '40px', fontSize: '16px' }}
        >
          Search for applicants
        </button>
      </div>
    </div>
  );
};

export default CompareSkillsPage;
