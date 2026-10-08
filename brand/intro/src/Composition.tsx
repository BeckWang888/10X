import { Composition } from "remotion";

export const MyComposition = () => {
  return (
    <Composition id="MyComp" component={MyComponent} durationInFrames={60} fps={30} width={320} height={180} />
  );
};

export const MyComponent: React.FC = () => {
  return <div style={{ flex: 1, background: "red", width: "100%", height: "100%" }} />;
};
