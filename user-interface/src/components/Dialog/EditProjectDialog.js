import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  Button,
  Input,
  Textarea,
  VStack,
} from '@chakra-ui/react';
import { Separator } from 'components/Separator/Separator';

function EditProjectDialog({ isOpen, onClose, project, onUpdate }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  useEffect(() => {
    if (project) {
      setName(project.name);
      setDescription(project.description);
    }
  }, [project]);

  const handleUpdate = () => {
    if (name.trim() && description.trim()) {
      onUpdate({ projectId: project.projectId, name, description });
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay bg="blackAlpha.800" />
      <ModalContent
        bg="linear-gradient(127.09deg, rgba(24, 29, 60, 0.94) 19.41%, rgba(10, 14, 35, 0.9) 76.65%)"
        borderRadius="15px"
        color="white"
      >
        <ModalHeader>Edit Project</ModalHeader>
        <Separator />
        <ModalCloseButton color="#fff" />
        <ModalBody>
          <VStack spacing={4}>
            <Input
              placeholder="Project Name"
              value={name}
              onChange={e => setName(e.target.value)}
              mb={4}
              borderColor="gray.600"
              bg="gray.800"
              _placeholder={{ color: 'gray.400' }}
              color="white"
              focusBorderColor="teal.400"
            />
            <Textarea
              placeholder="Project Description"
              value={description}
              onChange={e => setDescription(e.target.value)}
              borderColor="gray.600"
              bg="gray.800"
              _placeholder={{ color: 'gray.400' }}
              color="white"
              focusBorderColor="teal.400"
            />
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button onClick={onClose} mr={3} variant="ghost" color="white">
            Cancel
          </Button>
          <Button
            colorScheme="teal"
            onClick={handleUpdate}
            isDisabled={!name.trim() || !description.trim()}
          >
            Update
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

export default EditProjectDialog;
