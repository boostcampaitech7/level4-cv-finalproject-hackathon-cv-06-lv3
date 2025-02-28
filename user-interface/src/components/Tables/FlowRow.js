import { Box, Flex, IconButton, Text, Tooltip } from '@chakra-ui/react';
import React, { useState } from 'react';
import { FaLock, FaUnlock, FaPencilAlt, FaTrashAlt } from 'react-icons/fa';

function FlowRow({ name: flowName, description, onEdit, onDelete, onClick }) {
  const [isLocked, setIsLocked] = useState(false);

  const handleLockToggle = () => {
    setIsLocked(prev => !prev);
  };

  return (
    <Box
      p="16px"
      bg="linear-gradient(127.09deg, rgba(24, 29, 60, 0.94) 19.41%, rgba(10, 14, 35, 0.49) 76.65%)"
      transition="background-color 0.3s ease"
      _hover={!isLocked ? { bg: 'rgba(6, 12, 41, 0.8)' } : undefined} // 잠금 상태일 때 hover 제거
      cursor="pointer"
      onClick={onClick} // Flow 클릭 이벤트
      borderRadius="20px"
      boxShadow="md"
      w="100%"
    >
      <Flex justify="space-between" alignItems="center" w="100%">
        {/* Flow 정보 */}
        <Flex direction="column" maxW="70%">
          <Tooltip label={flowName} aria-label="Flow Name Tooltip">
            <Text color="#fff" fontSize="lg" fontWeight="bold" isTruncated>
              {flowName}
            </Text>
          </Tooltip>
        </Flex>

        {/* 액션 버튼 */}
        <Flex gap="10px" align="center">
          {/* 잠금 버튼 */}
          <IconButton
            size="sm"
            color={isLocked ? 'yellow.400' : 'gray.500'}
            bg="transparent"
            icon={isLocked ? <FaLock /> : <FaUnlock />}
            onClick={e => {
              e.stopPropagation();
              handleLockToggle();
            }}
            aria-label="Lock or Unlock Flow"
          />
          {/* 수정 버튼 */}
          <IconButton
            size="sm"
            color="green.500"
            bg="transparent"
            icon={<FaPencilAlt />}
            onClick={e => {
              e.stopPropagation();
              onEdit();
            }}
            aria-label="Edit Flow"
            isDisabled={isLocked} // 잠금 상태일 때 비활성화
            _hover={isLocked ? { color: 'green.400' } : undefined} // 잠금 상태에서 hover 제거
          />
          {/* 삭제 버튼 */}
          <IconButton
            size="sm"
            color="red.500"
            bg="transparent"
            icon={<FaTrashAlt />}
            onClick={e => {
              e.stopPropagation();
              if (!window.confirm('Are you sure you want to delete this flow?')) {
                return;
              }
              onDelete();
            }}
            aria-label="Delete Flow"
            isDisabled={isLocked} // 잠금 상태일 때 비활성화
            _hover={isLocked ? { color: 'red.400' } : undefined} // 잠금 상태에서 hover 제거
          />
        </Flex>
      </Flex>
    </Box>
  );
}

export default FlowRow;
