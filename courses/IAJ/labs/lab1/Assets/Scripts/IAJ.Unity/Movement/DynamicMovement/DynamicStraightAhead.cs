namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public class DynamicStraightAhead : DynamicMovement
    {

        public override string Name
        {
            get { return "StraightAhead"; }
        }

        public DynamicStraightAhead()
        {
            this.Output = new MovementOutput();
        }


        public override MovementOutput GetMovement()
        {

            this.Output.linear = this.Character.KinematicData.GetOrientationAsVector();

            if (this.Output.linear.sqrMagnitude > 0)
            {
                this.Output.linear.Normalize();
                this.Output.linear *= this.Character.MaxAcceleration;
            }

            return this.Output;
        }
    }
}
