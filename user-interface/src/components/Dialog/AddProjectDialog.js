import React, { useState } from 'react';
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
  Flex,
} from '@chakra-ui/react';

function AddProjectDialog({ isOpen, onClose, onAdd }) {
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');

  const handleAdd = () => {
    if (newProjectName.trim() && newProjectDescription.trim()) {
      onAdd({ name: newProjectName, description: newProjectDescription });
      setNewProjectName('');
      setNewProjectDescription('');
      onClose();
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="md" isCentered>
      <ModalOverlay bg="blackAlpha.800" />
      <ModalContent bg="gray.900" color="white" borderRadius="md">
        <ModalHeader>Add New Project</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4}>
            <Input
              placeholder="Project Name"
              value={newProjectName}
              onChange={e => setNewProjectName(e.target.value)}
              focusBorderColor="teal.400"
              bg="gray.800"
              borderRadius="md"
              color="white"
              _placeholder={{ color: 'gray.400' }}
            />
            <Textarea
              placeholder="Project Description"
              value={newProjectDescription}
              onChange={e => setNewProjectDescription(e.target.value)}
              focusBorderColor="teal.400"
              bg="gray.800"
              borderRadius="md"
              color="white"
              _placeholder={{ color: 'gray.400' }}
            />
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Flex w="100%" justifyContent="flex-end">
            <Button onClick={onClose} mr={3} variant="ghost" color="white">
              Cancel
            </Button>
            <Button
              colorScheme="teal"
              onClick={handleAdd}
              disabled={!newProjectName.trim() || !newProjectDescription.trim()}
            >
              Add
            </Button>
          </Flex>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

export default AddProjectDialog;
