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
  VStack,
} from '@chakra-ui/react';

function AddFlowDialog({ isOpen, onClose, onAdd }) {
  const [flowName, setFlowName] = useState('');

  const handleAdd = () => {
    if (flowName.trim()) {
      onAdd({ name: flowName });
      setFlowName('');
      onClose();
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay bg="blackAlpha.800" />
      <ModalContent bg="gray.900" color="white" borderRadius="md">
        <ModalHeader>Add New Flow</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4}>
            <Input
              placeholder="Flow Name"
              value={flowName}
              onChange={e => setFlowName(e.target.value)}
              focusBorderColor="teal.400"
              bg="gray.800"
              borderRadius="md"
              color="white"
            />
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button colorScheme="teal" mr={3} onClick={handleAdd}>
            Add Flow
          </Button>
          <Button variant="ghost" onClick={onClose} color="white">
            Cancel
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

export default AddFlowDialog;
