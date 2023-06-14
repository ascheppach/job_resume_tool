import React, { useEffect, useState } from 'react';

const CompareSkillsPage = () => {
  const [imageData, setImageData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/compare_skills')
      .then(response => {
        if (response.ok) {
          return response.blob();
        }
        throw new Error('Error retrieving image');
      })
      .then(blob => {
        setImageData(URL.createObjectURL(blob));
      })
      .catch(error => {
        setError(error.message);
      });
  }, []);

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!imageData) {
    return <div>Loading image...</div>;
  }

  return (
    <div>
      <h1>Image Display</h1>
      <img src={imageData} alt="Image" />
    </div>
  );
};

export default CompareSkillsPage;