// components/DragAndDropArea.js
import React from 'react';
import { Icon, Text } from '@chakra-ui/react';
import Card from 'components/Card/Card.js';
import CardBody from 'components/Card/CardBody.js';
import { useDropzone } from 'react-dropzone';
import { CartIcon } from 'components/Icons/Icons.js';

const DragAndDropArea = ({ onFilesAdded }) => {
  const onDrop = acceptedFiles => {
    console.log('Dropped files:', acceptedFiles);
    onFilesAdded(acceptedFiles);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <Card
      {...getRootProps()}
      borderRadius="md"
      textAlign="center"
      cursor="pointer"
      transition="background-color 0.3s ease"
      w="100%"
      h="100%"
      overflow="hidden"
      _hover={{ bg: 'rgba(6, 12, 41, 0.8)' }}
    >
      <CardBody
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        h="100%"
      >
        <Icon as={CartIcon} w={8} h={8} color="#fff" mb={2} />
        <input {...getInputProps()} />
        {isDragActive ? (
          <Text color="#fff" fontSize="sm" fontWeight="bold">
            Drop here
          </Text>
        ) : (
          <Text color="#fff" fontSize="sm" fontWeight="bold">
            Drop the file here or click here to upload a file.
          </Text>
        )}
      </CardBody>
    </Card>
  );
};

export default DragAndDropArea;
