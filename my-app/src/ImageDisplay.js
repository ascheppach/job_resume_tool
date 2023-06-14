import React, { Component } from 'react';

class ImageDisplay extends Component {
  constructor(props) {
    super(props);
    this.state = {
      imageData: null,
      error: null
    };
  }

  componentDidMount() {
    fetch('/compare_skills')
      .then(response => {
        if (response.ok) {
          return response.blob();
        }
        throw new Error('Error retrieving image');
      })
      .then(blob => {
        this.setState({ imageData: URL.createObjectURL(blob) });
      })
      .catch(error => {
        this.setState({ error: error.message });
      });
  }

  componentWillUnmount() {
    if (this.state.imageData) {
      URL.revokeObjectURL(this.state.imageData);
    }
  }

  render() {
    const { imageData, error } = this.state;

    if (error) {
      return <div>Error: {error}</div>;
    }

    if (!imageData) {
      return <div>Loading image...</div>;
    }

    return (
      <div>
        <h1>My Image</h1>
        <img src={imageData} alt="My Image" />
      </div>
    );
  }
}

export default ImageDisplay;