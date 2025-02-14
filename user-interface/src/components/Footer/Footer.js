/*eslint-disable*/
import React from 'react';
import { Flex, Link, List, ListItem, Text } from '@chakra-ui/react';

export default function Footer(props) {
  return (
    <Flex
      flexDirection={{
        base: 'column',
        xl: 'row',
      }}
      alignItems={{
        base: 'center',
        xl: 'start',
      }}
      justifyContent="space-between"
      px="30px"
      pb="20px"
    >
      <Text
        fontSize="sm"
        color="white"
        textAlign={{
          base: 'center',
          xl: 'start',
        }}
        mb={{ base: '20px', xl: '0px' }}
      >
        &copy; {1900 + new Date().getYear()},{' '}
        <Text as="span">{'Made ❤️ by '}</Text>
        <Link
          href="https://wise-columnist-5eb.notion.site/CV06-Hand-bone-Semantic-Segmentation-162218e09e1680eca386e32c28e46799?pvs=4"
          target="_blank"
        >
          {' SIXSENSE'}
        </Link>
        {' for prescriptive AI'}
      </Text>
      <List display="flex">
        <ListItem
          me={{
            base: '20px',
            md: '44px',
          }}
        >
          <Link
            color="white"
            fontSize="sm"
            href="https://github.com/boostcampaitech7/level4-cv-finalproject-hackathon-cv-06-lv3.git"
          >
            {'GitHub'}
          </Link>
        </ListItem>
        <ListItem
          me={{
            base: '20px',
            md: '44px',
          }}
        >
          <Link
            color="white"
            fontSize="sm"
            href="https://www.instagram.com/cv06_sixsense/"
          >
            {'Instagram'}
          </Link>
        </ListItem>
        <ListItem
          me={{
            base: '20px',
            md: '44px',
          }}
        >
          <Link
            color="white"
            fontSize="sm"
            href="https://wise-columnist-5eb.notion.site/CV06-Hand-bone-Semantic-Segmentation-162218e09e1680eca386e32c28e46799?pvs=4"
          >
            {'Notion'}
          </Link>
        </ListItem>
      </List>
    </Flex>
  );
}
