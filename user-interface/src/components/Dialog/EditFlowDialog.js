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
  VStack,
} from '@chakra-ui/react';

function EditFlowDialog({ isOpen, onClose, flow, onUpdate }) {
  const [flowName, setFlowName] = useState('');

  useEffect(() => {
    if (flow) {
      setFlowName(flow.flow_name || '');
    }
  }, [flow]);

  const handleUpdate = () => {
    if (flowName.trim()) {
      console.log('✅ Updating flow:', { flowId: flow.flowId, flowName });
      onUpdate(flow.flowId, flowName);
      onClose();
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay bg="blackAlpha.800" />
      <ModalContent bg="gray.900" color="white" borderRadius="md">
        <ModalHeader>Edit Flow</ModalHeader>
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
          <Button colorScheme="teal" mr={3} onClick={handleUpdate}>
            Update Flow
          </Button>
          <Button variant="ghost" onClick={onClose} color="white">
            Cancel
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

export default EditFlowDialog;
