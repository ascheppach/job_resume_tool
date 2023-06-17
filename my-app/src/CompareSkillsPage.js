import React, { useState, useRef } from 'react';

const CompareSkillsPage = () => {
  const [skill, setSkill] = useState('');
  const [skills, setSkills] = useState([]);
  const [currentSkill, setCurrentSkill] = useState('');
  const inputRef2 = useRef(null);

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
    console.log('Skill:', skill);
    console.log('Corresponding skill catalogue:', skills);
    setSkills([]);
  };

  return (
    <div>
      <h1>Compare Skills Page</h1>

      <form onSubmit={handleSkillSubmit}>
        <div>
          <label>
            Skill:
            <input
              type="text"
              value={skill}
              onChange={handleSkillChange}
              placeholder="Enter a skill..."
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
    </div>
  );
};

export default CompareSkillsPage;
